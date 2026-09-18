import type { ArrhythmiaEvent, ArrhythmiaEventType, HRVData, RPeak } from '../types';

/**
 * 前端本地分析与后端 ecg_service 共用同一套判定口径，
 * 保证“分析页面（状态栏/摘要）”与“结果面板（事件列表）”结论一致。
 *
 * 关键约定：
 * - 有效心跳不足（R 峰 < 3）时，heartRate 返回 0，判定结果只能是
 *   insufficient_data（数据不足），不得判为正常或任何异常。
 * - ST 段测量窗口随各拍 RR 间期缩放，避免高心率时固定时间窗口
 *   落到 T 波上把普通波形误报为 ST 抬高。
 */

/** 可用于 HRV/心律判定的最少心跳数（至少 2 个 RR 间期） */
export const MIN_R_PEAKS = 3;

/**
 * 居中滑动平均（零相位近似），边界处用可用样本归一化。
 */
function movingAverage(input: ArrayLike<number>, width: number): Float64Array {
  const out = new Float64Array(input.length);
  let run = 0;
  for (let i = 0; i < input.length; i++) {
    run += input[i];
    if (i >= width) run -= input[i - width];
    out[i] = run / Math.min(i + 1, width);
  }
  return out;
}

/**
 * 简化版 Pan-Tompkins R 峰检测：
 * 去基线（300ms 滑动均值）→ 带通（10~30ms 均值差）→ 微分 → 平方 →
 * 80ms 滑动积分 → 自适应阈值选峰。
 *
 * 与原始“只看正幅度”阈值法相比：
 * - 不依赖主波方向：带通后正负向 QRS（如 aVR）都能检出；
 * - 平坦/纯噪声信号检不出峰（带通能量受噪声底门控），
 *   不会把噪声点误数成心跳再“算个心率出来”；
 * - 每段连续超阈值区间只取能量最大处，且施加 200ms 不应期，
 *   不会把一个 QRS 数成两拍。
 */
export function detectRPeaks(samples: number[], sr: number): RPeak[] {
  const n = samples.length;
  if (n < Math.ceil(0.5 * sr)) return [];

  // 1. 去基线漂移（~300ms 低通后从原信号减去）
  const baseline = movingAverage(samples, Math.round(0.3 * sr));
  const hp = new Float64Array(n);
  for (let i = 0; i < n; i++) hp[i] = samples[i] - baseline[i];

  // 2. 带通（短窗均值 - 长窗均值），压低 T/P 波与高频噪声
  const shortWin = Math.max(3, Math.round(0.01 * sr));
  const longWin = Math.max(6, Math.round(0.03 * sr));
  const avgShort = movingAverage(hp, shortWin);
  const avgLong = movingAverage(hp, longWin);
  const bp = new Float64Array(n);
  for (let i = 0; i < n; i++) bp[i] = avgShort[i] - avgLong[i];

  // 3. 微分 + 4. 平方
  const squared = new Float64Array(n);
  for (let i = 1; i < n - 1; i++) {
    const d = (bp[i + 1] - bp[i - 1]) * 0.5;
    squared[i] = d * d;
  }

  // 5. 滑动窗口积分（80ms：覆盖 QRS 宽度，又不会在高心率时连到下一拍）
  const integWin = Math.round(0.08 * sr);
  const integrated = movingAverage(squared, integWin);

  // 自适应阈值：均值 + 1 个标准差
  let mean = 0;
  for (let i = 0; i < n; i++) mean += integrated[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (integrated[i] - mean) ** 2;
  const std = Math.sqrt(variance / n);
  const threshold = mean + std;

  // 带通信号的噪声底：对中位数的 MAD（稳健，不被 QRS 少数大值抬高）
  const sortedBp = Float64Array.from(bp).sort();
  const medianBp = sortedBp[Math.floor(n / 2)];
  const deviations = new Float64Array(n);
  for (let i = 0; i < n; i++) deviations[i] = Math.abs(bp[i] - medianBp);
  deviations.sort();
  const noiseFloor = deviations[Math.floor(n / 2)] / 0.6745 || 1e-9;

  const minDistance = Math.round(0.2 * sr);
  const rPeaks: RPeak[] = [];
  let lastBeat = -minDistance;
  let segmentStart = -1;

  // 每一段连续超阈值区间代表一个 QRS 候选：取区间内积分能量最大处
  for (let i = 0; i <= n; i++) {
    if (i < n && integrated[i] > threshold) {
      if (segmentStart < 0) segmentStart = i;
      continue;
    }
    if (segmentStart < 0) continue;

    let beatIndex = segmentStart;
    let bestEnergy = -1;
    for (let j = segmentStart; j < i; j++) {
      if (integrated[j] > bestEnergy) {
        bestEnergy = integrated[j];
        beatIndex = j;
      }
    }
    segmentStart = -1;
    if (beatIndex - lastBeat < minDistance) continue;

    // 在带通信号 ±80ms 窗口定位真实偏转，并要求显著高于噪声底
    const lo = Math.max(0, beatIndex - Math.round(0.08 * sr));
    const hi = Math.min(n, beatIndex + Math.round(0.08 * sr));
    let peakIndex = lo;
    let peakAmplitude = 0;
    for (let j = lo; j < hi; j++) {
      const mag = Math.abs(bp[j]);
      if (mag > peakAmplitude) {
        peakAmplitude = mag;
        peakIndex = j;
      }
    }
    if (peakAmplitude < 5 * noiseFloor) continue;

    rPeaks.push({
      index: peakIndex,
      time: peakIndex / sr,
      amplitude: samples[peakIndex],
    });
    lastBeat = beatIndex;
  }

  return rPeaks;
}

/**
 * 计算 HRV 指标。
 * R 峰不足时 heartRate 为 0、各指标为 0，由上层据此输出“数据不足”，
 * 不再回退到模拟心率设定值（否则会与实际数到的心跳对不上）。
 */
export function calculateHRV(rPeaks: RPeak[], sr: number): HRVData {
  if (rPeaks.length < MIN_R_PEAKS) {
    return { heartRate: 0, sdnn: 0, rmssd: 0, pnn50: 0, nnIntervals: [] };
  }

  const nnIntervals: number[] = [];
  for (let i = 1; i < rPeaks.length; i++) {
    nnIntervals.push(((rPeaks[i].index - rPeaks[i - 1].index) / sr) * 1000);
  }

  const meanRR = nnIntervals.reduce((a, b) => a + b, 0) / nnIntervals.length;
  const hr = meanRR > 0 ? 60000 / meanRR : 0;

  const variance = nnIntervals.reduce((sum, x) => sum + (x - meanRR) ** 2, 0) / nnIntervals.length;
  const sdnn = Math.sqrt(variance);

  let sumSquaredDiffs = 0;
  for (let i = 1; i < nnIntervals.length; i++) {
    sumSquaredDiffs += (nnIntervals[i] - nnIntervals[i - 1]) ** 2;
  }
  const rmssd = Math.sqrt(sumSquaredDiffs / (nnIntervals.length - 1));

  let nn50Count = 0;
  for (let i = 1; i < nnIntervals.length; i++) {
    if (Math.abs(nnIntervals[i] - nnIntervals[i - 1]) > 50) nn50Count++;
  }
  const pnn50 = (nn50Count / (nnIntervals.length - 1)) * 100;

  return {
    heartRate: Math.round(hr * 10) / 10,
    sdnn: Math.round(sdnn * 100) / 100,
    rmssd: Math.round(rmssd * 100) / 100,
    pnn50: Math.round(pnn50 * 100) / 100,
    nnIntervals,
  };
}

/** ST 抬高判定阈值 (mV)：相对等电位基线的抬升幅度 */
const ST_ELEVATION_THRESHOLD = 0.1;

/** 需要至少多少个可测量的心拍才允许判读 ST 段 */
const ST_MIN_VALID_BEATS = 3;

/**
 * 基于 R 峰附近的等电位段估计该拍基线。
 * 取 R 峰前 0.30~0.24 个 RR 周期（即前一拍 T 波结束后的 TP 段），
 * 覆盖跨周期边界；窗口长度固定 0.06 个 RR，各心率下都有足够采样点。
 */
function beatBaseline(samples: number[], idx: number, rrSamples: number, sr: number): number | null {
  const width = Math.max(4, Math.floor(0.06 * rrSamples));
  const end = idx - Math.floor(0.24 * rrSamples);
  const start = end - width;
  if (end <= 0) return null;
  const lo = Math.max(0, start);
  if (end - lo < 3) return null;
  let sum = 0;
  for (let j = lo; j < end; j++) sum += samples[j];
  return sum / (end - lo);
}

/**
 * 统计单个心拍的 ST 段相对基线抬升 (mV)。
 * ST 段取 R 峰后 0.06~0.10 个 RR 周期（约 J 点+60ms 附近，随心率缩放），
 * 正常 T 波上升支在 0.14 RR 之后才开始，因此不会把 T 波误当成抬高。
 * 无法测量（信号截断/窗口不足）时返回 null，该拍不计入任何判读。
 */
function measureSTElevation(
  samples: number[],
  idx: number,
  rrSamples: number,
  sr: number,
): number | null {
  const stStart = idx + Math.floor(0.06 * rrSamples);
  const stEnd = idx + Math.floor(0.10 * rrSamples);
  if (stEnd >= samples.length) return null;
  const baseline = beatBaseline(samples, idx, rrSamples, sr);
  if (baseline === null) return null;
  let stSum = 0;
  for (let j = stStart; j < stEnd; j++) stSum += samples[j];
  return stSum / (stEnd - stStart) - baseline;
}

/**
 * 心律失常检测。判定顺序与后端 detect_arrhythmia 完全一致。
 */
export function detectArrhythmias(
  hrv: HRVData,
  rPeaks: RPeak[],
  samples: number[],
  sr: number,
): ArrhythmiaEvent[] {
  // 心跳不足：单独说明数据不足，不判正常也不判异常
  if (rPeaks.length < MIN_R_PEAKS || hrv.heartRate <= 0) {
    return [
      {
        eventType: 'insufficient_data',
        confidence: 1.0,
        description: `有效心跳不足（检测到 ${rPeaks.length} 次，至少需要 ${MIN_R_PEAKS} 次），数据不足以判读心律`,
        timestamp: rPeaks[0]?.time ?? 0,
      },
    ];
  }

  const events: ArrhythmiaEvent[] = [];
  const hr = hrv.heartRate;

  if (hr > 100) {
    events.push({
      eventType: 'tachycardia',
      confidence: Math.min(1.0, (hr - 100) / 50 + 0.6),
      description: `心率过快 (${hr.toFixed(0)} BPM)，检测到心动过速`,
      timestamp: rPeaks[0].time,
    });
  }

  if (hr < 60) {
    events.push({
      eventType: 'bradycardia',
      confidence: Math.min(1.0, (60 - hr) / 30 + 0.6),
      description: `心率过慢 (${hr.toFixed(0)} BPM)，检测到心动过缓`,
      timestamp: rPeaks[0].time,
    });
  }

  // ST 段抬高：逐拍用各自的 RR 间期自适应测量，超半数可测量心拍抬高才报警
  const rrSamples = hrv.nnIntervals.map((rr) => (rr / 1000) * sr);
  let elevatedCount = 0;
  let measuredCount = 0;
  // 最后一拍没有后续 RR，沿用前一拍的 RR
  for (let i = 0; i < rPeaks.length; i++) {
    const rr = rrSamples[Math.min(i, rrSamples.length - 1)];
    const elevation = measureSTElevation(samples, rPeaks[i].index, rr, sr);
    if (elevation === null) continue;
    measuredCount++;
    if (elevation > ST_ELEVATION_THRESHOLD) elevatedCount++;
  }
  if (measuredCount >= ST_MIN_VALID_BEATS && elevatedCount > measuredCount * 0.5) {
    events.push({
      eventType: 'st_elevation',
      confidence: Math.min(1.0, elevatedCount / measuredCount),
      description: '检测到 ST 段抬高，可能提示心肌梗死',
      timestamp: rPeaks[0].time,
    });
  }

  // 心律不规则（变异系数 > 0.15）
  if (hrv.nnIntervals.length > 3) {
    const mean = hrv.nnIntervals.reduce((a, b) => a + b, 0) / hrv.nnIntervals.length;
    const variance = hrv.nnIntervals.reduce((s, x) => s + (x - mean) ** 2, 0) / hrv.nnIntervals.length;
    const cv = mean > 0 ? Math.sqrt(variance) / mean : 0;
    if (cv > 0.15) {
      events.push({
        eventType: 'atrial_fibrillation',
        confidence: Math.min(1.0, cv * 2),
        description: 'RR 间期不规则，可能提示房颤',
        timestamp: rPeaks[0].time,
      });
    }
  }

  if (events.length === 0) {
    events.push({
      eventType: 'normal',
      confidence: 1.0,
      description: '正常窦性心律',
      timestamp: rPeaks[0].time,
    });
  }

  return events;
}

/**
 * 汇总诊断文案。与后端 get_rhythm_diagnosis 保持一致，
 * 状态栏、结果面板都从同一份事件 + HRV 推导，口径不会再分叉。
 */
export function buildRhythmDiagnosis(events: ArrhythmiaEvent[], hrv: HRVData): string {
  const types = events.map((e) => e.eventType);

  if (types.includes('insufficient_data')) {
    return `数据不足，无法判读心律 | 有效心跳: ${events[0] ? extractBeatCount(events[0].description) : 0} 次 | 心率无法计算`;
  }
  if (types.includes('st_elevation')) return 'ST 段抬高 - 建议立即就医检查';
  if (types.includes('tachycardia') && types.includes('atrial_fibrillation')) {
    return '快速房颤 - 建议进一步心脏评估';
  }
  if (types.includes('tachycardia')) return '窦性心动过速 - 请结合临床症状判断';
  if (types.includes('bradycardia')) return '窦性心动过缓 - 建议关注心率变化';
  if (types.includes('atrial_fibrillation')) return '心律不规则 - 疑似房颤，建议 Holter 监测';

  const hr = hrv.heartRate;
  return `正常窦性心律 | HR: ${hr.toFixed(0)} BPM | SDNN: ${hrv.sdnn.toFixed(1)} ms`;
}

function extractBeatCount(description: string): number {
  const match = description.match(/检测到\s*(\d+)\s*次/);
  return match ? Number(match[1]) : 0;
}

/** 事件类型 → 面板中文标签（App.vue 使用） */
export const EVENT_LABELS: Record<ArrhythmiaEventType, string> = {
  normal: '正常窦性心律',
  tachycardia: '心动过速',
  bradycardia: '心动过缓',
  st_elevation: 'ST 段抬高',
  atrial_fibrillation: '房颤',
  premature_ventricular_contraction: '室性早搏',
  insufficient_data: '数据不足',
};
