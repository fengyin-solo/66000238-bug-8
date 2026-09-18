import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import type { ECGLead, HRVData, RPeak, ArrhythmiaEvent, ECGAnalysisResponse } from '../types';
import { MIN_BEATS_FOR_ANALYSIS, DIAGNOSIS_INSUFFICIENT_DATA } from '../types';

// Gaussian function for PQRST wave simulation
function gaussian(x: number, amplitude: number, center: number, width: number): number {
  return amplitude * Math.exp(-((x - center) ** 2) / (2 * width ** 2));
}

// Lead-specific PQRST configuration
interface LeadConfig {
  pAmplitude: number;
  qAmplitude: number;
  rAmplitude: number;
  sAmplitude: number;
  tAmplitude: number;
  stElevation: number;
}

const LEAD_CONFIGS: Record<string, LeadConfig> = {
  'I': { pAmplitude: 0.12, qAmplitude: -0.05, rAmplitude: 0.8, sAmplitude: -0.1, tAmplitude: 0.25, stElevation: 0.0 },
  'II': { pAmplitude: 0.15, qAmplitude: -0.1, rAmplitude: 1.2, sAmplitude: -0.2, tAmplitude: 0.3, stElevation: 0.0 },
  'III': { pAmplitude: 0.10, qAmplitude: -0.08, rAmplitude: 0.9, sAmplitude: -0.15, tAmplitude: 0.2, stElevation: 0.0 },
  'aVR': { pAmplitude: -0.10, qAmplitude: 0.05, rAmplitude: -0.8, sAmplitude: 0.1, tAmplitude: -0.2, stElevation: 0.0 },
  'aVL': { pAmplitude: 0.10, qAmplitude: -0.03, rAmplitude: 0.6, sAmplitude: -0.05, tAmplitude: 0.2, stElevation: 0.0 },
  'aVF': { pAmplitude: 0.13, qAmplitude: -0.09, rAmplitude: 1.0, sAmplitude: -0.18, tAmplitude: 0.28, stElevation: 0.0 },
  'V1': { pAmplitude: 0.08, qAmplitude: 0.0, rAmplitude: 0.3, sAmplitude: -0.8, tAmplitude: 0.15, stElevation: 0.0 },
  'V2': { pAmplitude: 0.10, qAmplitude: -0.02, rAmplitude: 0.6, sAmplitude: -0.6, tAmplitude: 0.25, stElevation: 0.0 },
  'V3': { pAmplitude: 0.10, qAmplitude: -0.05, rAmplitude: 0.9, sAmplitude: -0.4, tAmplitude: 0.3, stElevation: 0.0 },
  'V4': { pAmplitude: 0.12, qAmplitude: -0.08, rAmplitude: 1.3, sAmplitude: -0.25, tAmplitude: 0.35, stElevation: 0.0 },
  'V5': { pAmplitude: 0.12, qAmplitude: -0.1, rAmplitude: 1.1, sAmplitude: -0.15, tAmplitude: 0.3, stElevation: 0.0 },
  'V6': { pAmplitude: 0.10, qAmplitude: -0.08, rAmplitude: 0.9, sAmplitude: -0.1, tAmplitude: 0.25, stElevation: 0.0 },
};

// ST measurement parameters — mirrored from the backend so both analysis
// paths apply the same rule.
const ST_ELEVATION_THRESHOLD_MV = 0.1;
const ST_BASELINE_MS: readonly [number, number] = [-65, -45];
const ST_LEVEL_MS: readonly [number, number] = [50, 100];
const MIN_QRS_AMPLITUDE_MV = 0.05;

// Generate a single PQRST cycle at normalized time t (0 to 1)
function generatePQRSTCycle(tNorm: number, config: LeadConfig): number {
  const p = gaussian(tNorm, config.pAmplitude, 0.12, 0.035);
  const q = gaussian(tNorm, config.qAmplitude, 0.22, 0.012);
  const r = gaussian(tNorm, config.rAmplitude, 0.26, 0.012);
  const s = gaussian(tNorm, config.sAmplitude, 0.30, 0.015);
  const tWave = gaussian(tNorm, config.tAmplitude, 0.48, 0.055);
  const st = (tNorm > 0.32 && tNorm < 0.42) ? config.stElevation : 0.0;
  return p + q + r + s + tWave + st;
}

/**
 * Overall rhythm diagnosis. This is the frontend counterpart of the backend
 * get_rhythm_diagnosis(): same priority, same wording. The status bar and the
 * result panel both read this string, so they can never diverge.
 */
function buildRhythmDiagnosis(events: ArrhythmiaEvent[], hrv: HRVData): string {
  const types = events.map((e) => e.eventType);

  if (types.includes('insufficient_data') || !hrv.dataSufficient || hrv.heartRate === null) {
    return DIAGNOSIS_INSUFFICIENT_DATA;
  }
  if (types.includes('st_elevation')) {
    return 'ST 段抬高 - 检测到 ST 段抬高，建议立即就医检查';
  }
  if (types.includes('tachycardia') && types.includes('atrial_fibrillation')) {
    return '快速房颤 - 建议进一步心脏评估';
  }
  if (types.includes('tachycardia')) {
    return '窦性心动过速 - 请结合临床症状判断';
  }
  if (types.includes('bradycardia')) {
    return '窦性心动过缓 - 建议关注心率变化';
  }
  if (types.includes('atrial_fibrillation')) {
    return '心律不规则 - 疑似房颤，建议 Holter 监测';
  }
  return `正常窦性心律 | HR: ${hrv.heartRate.toFixed(0)} BPM | SDNN: ${(hrv.sdnn ?? 0).toFixed(1)} ms`;
}

export const useECGStore = defineStore('ecg', () => {
  // State
  const selectedLead = ref<string>('II');
  const heartRate = ref<number>(72);
  const samplingRate = ref<number>(500);
  const duration = ref<number>(10);
  const isMonitoring = ref<boolean>(false);
  const ecgData = ref<ECGLead | null>(null);
  const hrvData = ref<HRVData | null>(null);
  const arrhythmiaEvents = ref<ArrhythmiaEvent[]>([]);
  const rhythmDiagnosis = ref<string>('');
  const isLoading = ref<boolean>(false);
  const useBackend = ref<boolean>(false);
  const backendUrl = ref<string>('http://localhost:8000');
  // Set when the backend analysis call fails; drives the retry banner.
  const apiError = ref<string | null>(null);

  let animationTimer: ReturnType<typeof setInterval> | null = null;
  const scrollOffset = ref<number>(0);

  // Getters
  const currentSamples = computed(() => ecgData.value?.samples ?? []);
  const currentRPeaks = computed(() => ecgData.value?.rPeaks ?? []);
  // The configured simulation rate is only an input; never show it as the
  // measured heart rate when beats could not be detected.
  const currentHeartRate = computed<number | null>(() => hrvData.value?.heartRate ?? null);
  const dataSufficient = computed<boolean>(() => hrvData.value?.dataSufficient ?? false);

  // Actions

  /**
   * Generate realistic 12-lead ECG waveform data with PQRST morphology
   */
  function generateECGWaveform(): ECGLead {
    const totalSamples = Math.floor(duration.value * samplingRate.value);
    const samples: number[] = new Array(totalSamples);
    const config = LEAD_CONFIGS[selectedLead.value] || LEAD_CONFIGS['II'];
    const cycleDuration = 60.0 / heartRate.value;

    for (let i = 0; i < totalSamples; i++) {
      const time = i / samplingRate.value;
      const cyclePosition = (time % cycleDuration) / cycleDuration;

      // Add slight HRV variation per beat
      const beatIndex = Math.floor(time / cycleDuration);
      const hrvFactor = 1.0 + Math.sin(beatIndex * 0.7) * 0.02;

      samples[i] = generatePQRSTCycle(cyclePosition, config) * hrvFactor;

      // Add baseline wander
      samples[i] += 0.03 * Math.sin(2 * Math.PI * 0.15 * time);
      // Add small noise
      samples[i] += (Math.random() - 0.5) * 0.02;
    }

    return {
      leadName: selectedLead.value,
      samplingRate: samplingRate.value,
      duration: duration.value,
      samples,
      rPeaks: [],
    };
  }

  /**
   * Pan-Tompkins style R-peak detection.
   *
   * Baseline removal -> squared derivative energy -> 150ms integration ->
   * local maxima above a *relative* threshold (200ms refractory gap) ->
   * +/-15ms polarity-aware refinement on the raw signal.
   *
   * The previous mean/std threshold returned nothing on flat traces and the
   * raw-signal refinement only looked for positive maxima, so inverted leads
   * (aVR) and deep-S leads (V1/V2) were misplaced on P/Q/T waves.
   */
  function detectRPeaks(samples: number[], sr: number): RPeak[] {
    const n = samples.length;
    if (n < 2) return [];

    // Simple centered moving average to remove baseline wander.
    // The window must be centered (zero-phase); a causal window shifts the
    // baseline and systematically displaces the detected peaks.
    const maWindow = Math.max(1, Math.floor(0.4 * sr));
    const baseline: number[] = new Array(n);
    const half = Math.floor(maWindow / 2);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - half);
      const hi = Math.min(n - 1, i + maWindow - half - 1);
      let sum2 = 0;
      for (let j = lo; j <= hi; j++) sum2 += samples[j];
      baseline[i] = sum2 / (hi - lo + 1);
    }
    const detrended = samples.map((s, i) => s - baseline[i]);

    // Light low-pass smoothing (approximates the 15 Hz upper cutoff of the
    // backend bandpass) before differentiation, to suppress muscle noise.
    const smoothed = detrended.map((v, i) => {
      const lo = Math.max(0, i - 1);
      const hi = Math.min(n - 1, i + 1);
      return (detrended[lo] + v + detrended[hi]) / (hi - lo + 1);
    });

    // Squared derivative, integrated over a *centered* 150ms window to match
    // the backend's zero-phase convolution.
    const windowSize = Math.max(1, Math.floor(0.15 * sr));
    const energy: number[] = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const d = (smoothed[Math.min(n - 1, i + 1)] - smoothed[Math.max(0, i - 1)]) / 2;
      energy[i] = d * d;
    }
    const integrated: number[] = new Array(n).fill(0);
    const wHalf = Math.floor(windowSize / 2);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - wHalf);
      const hi = Math.min(n - 1, i + windowSize - wHalf - 1);
      let sum2 = 0;
      for (let j = lo; j <= hi; j++) sum2 += energy[j];
      integrated[i] = sum2 / (hi - lo + 1);
    }

    const peakEnergy = Math.max(...integrated);
    if (peakEnergy <= 0) return [];
    // Relative threshold: low-amplitude leads (e.g. aVL) must still be found.
    const threshold = 0.2 * peakEnergy;
    const minDistance = Math.floor(0.2 * sr); // 200ms refractory gap
    // Signed-extremum refinement radius around the QRS energy center.
    const refineRadius = Math.floor(0.04 * sr); // +/-40ms
    // Local-maxima neighborhood: narrower than the integration window so
    // beats at high heart rates (close QRS complexes) aren't merged away.
    const localWindow = Math.max(1, Math.floor(0.1 * sr));

    const peaks: RPeak[] = [];
    const used = new Set<number>();
    const centers: number[] = [];

    for (let i = localWindow; i < n - localWindow; i++) {
      if (integrated[i] <= threshold) continue;

      // Local maximum within the local-maxima neighborhood
      let isLocalMax = true;
      for (let j = i - localWindow; j <= i + localWindow; j++) {
        if (j !== i && integrated[j] > integrated[i]) {
          isLocalMax = false;
          break;
        }
      }
      if (!isLocalMax) continue;
      if (centers.some((c) => Math.abs(c - i) < minDistance)) continue;
      centers.push(i);
    }

    if (centers.length === 0) return peaks;

    // Determine overall R polarity from the first detected QRS complexes:
    // the energy center is not the R peak, and complexes can be positive
    // (lead II), inverted (aVR) or S-dominated (V1/V2).
    const polarityRadius = Math.floor(0.05 * sr);
    const probes: number[] = [];
    for (const c of centers.slice(0, 6)) {
      const lo = Math.max(0, c - polarityRadius);
      const hi = Math.min(n - 1, c + polarityRadius);
      let maxV = -Infinity;
      let minV = Infinity;
      for (let j = lo; j <= hi; j++) {
        if (detrended[j] > maxV) maxV = detrended[j];
        if (detrended[j] < minV) minV = detrended[j];
      }
      probes.push(Math.abs(maxV) >= Math.abs(minV) ? maxV : minV);
    }
    const sortedProbes = [...probes].sort((a, b) => a - b);
    const medianProbe = sortedProbes[Math.floor(sortedProbes.length / 2)];
    const positivePolarity = medianProbe >= 0;

    // Refine to the signed extremum within +/-40ms of each QRS center.
    for (const center of centers) {
      const lo = Math.max(0, center - refineRadius);
      const hi = Math.min(n - 1, center + refineRadius);
      let idx = lo;
      for (let j = lo; j <= hi; j++) {
        if (positivePolarity ? detrended[j] > detrended[idx] : detrended[j] < detrended[idx]) {
          idx = j;
        }
      }

      if (!used.has(idx) && Math.abs(samples[idx]) >= MIN_QRS_AMPLITUDE_MV) {
        used.add(idx);
        peaks.push({
          index: idx,
          time: idx / sr,
          amplitude: samples[idx],
        });
      }
    }

    return peaks;
  }

  /**
   * Calculate HRV metrics from R-peak positions.
   * SDNN, RMSSD, pNN50.
   *
   * With fewer than MIN_BEATS_FOR_ANALYSIS beats, every metric is null and
   * dataSufficient is false: an unmeasurable heart rate is "unknown", not 0.
   */
  function calculateHRV(rPeaks: RPeak[], sr: number): HRVData {
    if (rPeaks.length < MIN_BEATS_FOR_ANALYSIS) {
      return {
        heartRate: null,
        sdnn: null,
        rmssd: null,
        pnn50: null,
        nnIntervals: [],
        beatCount: rPeaks.length,
        dataSufficient: false,
      };
    }

    const nnIntervals: number[] = [];
    for (let i = 1; i < rPeaks.length; i++) {
      const rr = ((rPeaks[i].index - rPeaks[i - 1].index) / sr) * 1000;
      nnIntervals.push(rr);
    }

    const meanRR = nnIntervals.reduce((a, b) => a + b, 0) / nnIntervals.length;
    const hr = meanRR > 0 ? 60000 / meanRR : null;

    // SDNN
    const variance = nnIntervals.reduce((sum2, x) => sum2 + (x - meanRR) ** 2, 0) / nnIntervals.length;
    const sdnn = Math.sqrt(variance);

    // RMSSD
    let sumSquaredDiffs = 0;
    for (let i = 1; i < nnIntervals.length; i++) {
      sumSquaredDiffs += (nnIntervals[i] - nnIntervals[i - 1]) ** 2;
    }
    const rmssd = Math.sqrt(sumSquaredDiffs / (nnIntervals.length - 1));

    // pNN50
    let nn50Count = 0;
    for (let i = 1; i < nnIntervals.length; i++) {
      if (Math.abs(nnIntervals[i] - nnIntervals[i - 1]) > 50) {
        nn50Count++;
      }
    }
    const pnn50 = (nn50Count / (nnIntervals.length - 1)) * 100;

    const round1 = (x: number) => Math.round(x * 10) / 10;
    return {
      heartRate: hr === null ? null : round1(hr),
      sdnn: Math.round(sdnn * 100) / 100,
      rmssd: Math.round(rmssd * 100) / 100,
      pnn50: Math.round(pnn50 * 100) / 100,
      nnIntervals,
      beatCount: rPeaks.length,
      dataSufficient: true,
    };
  }

  /**
   * Arrhythmia detection: tachycardia, bradycardia, ST-elevation.
   *
   * Insufficient data short-circuits to a single informational event and no
   * normal/abnormal conclusion is produced.
   */
  function detectArrhythmias(hrv: HRVData, rPeaks: RPeak[], samples: number[], sr: number): ArrhythmiaEvent[] {
    const events: ArrhythmiaEvent[] = [];
    const ts = rPeaks[0]?.time ?? 0;

    if (!hrv.dataSufficient || hrv.heartRate === null) {
      events.push({
        eventType: 'insufficient_data',
        confidence: 1.0,
        description: `有效心跳仅 ${rPeaks.length} 次，少于分析所需的 ${MIN_BEATS_FOR_ANALYSIS} 次，数据不足，无法判定心律是否正常`,
        timestamp: ts,
      });
      return events;
    }

    const hr = hrv.heartRate;

    if (hr > 100) {
      events.push({
        eventType: 'tachycardia',
        confidence: Math.min(1.0, (hr - 100) / 50 + 0.6),
        description: `心率过快 (${hr.toFixed(0)} BPM)，检测到心动过速`,
        timestamp: ts,
      });
    }

    if (hr < 60) {
      events.push({
        eventType: 'bradycardia',
        confidence: Math.min(1.0, (60 - hr) / 30 + 0.6),
        description: `心率过慢 (${hr.toFixed(0)} BPM)，检测到心动过缓`,
        timestamp: ts,
      });
    }

    // ST-segment elevation: compare the post-J-point ST segment against the
    // isoelectric PR segment. Both windows mirror the backend values, so a
    // normal PQRST wave (P/Q/T waves) is never mistaken for elevation.
    let stElevationCount = 0;
    let evaluatedCount = 0;
    for (const rp of rPeaks) {
      const blStart = rp.index + Math.floor((ST_BASELINE_MS[0] / 1000) * sr);
      const blEnd = rp.index + Math.floor((ST_BASELINE_MS[1] / 1000) * sr);
      const stStart = rp.index + Math.floor((ST_LEVEL_MS[0] / 1000) * sr);
      const stEnd = rp.index + Math.floor((ST_LEVEL_MS[1] / 1000) * sr);
      if (blStart < 0 || stEnd >= samples.length || stEnd <= stStart || blEnd <= blStart) continue;

      const stLevel = samples.slice(stStart, stEnd).reduce((a, b) => a + b, 0) / (stEnd - stStart);
      const baseline = samples.slice(blStart, blEnd).reduce((a, b) => a + b, 0) / (blEnd - blStart);
      evaluatedCount++;
      if (stLevel - baseline > ST_ELEVATION_THRESHOLD_MV) {
        stElevationCount++;
      }
    }

    if (evaluatedCount >= MIN_BEATS_FOR_ANALYSIS && stElevationCount > evaluatedCount * 0.5) {
      events.push({
        eventType: 'st_elevation',
        confidence: Math.min(1.0, stElevationCount / evaluatedCount),
        description: '检测到 ST 段抬高，可能提示心肌梗死，建议立即就医检查',
        timestamp: ts,
      });
    }

    if (events.length === 0) {
      events.push({
        eventType: 'normal',
        confidence: 1.0,
        description: '正常窦性心律',
        timestamp: ts,
      });
    }

    return events;
  }

  /**
   * Run full ECG analysis (frontend simulation)
   */
  function runFrontendAnalysis() {
    const lead = generateECGWaveform();
    const peaks = detectRPeaks(lead.samples, lead.samplingRate);
    lead.rPeaks = peaks;
    ecgData.value = lead;

    const hrv = calculateHRV(peaks, lead.samplingRate);
    hrvData.value = hrv;

    const events = detectArrhythmias(hrv, peaks, lead.samples, lead.samplingRate);
    arrhythmiaEvents.value = events;
    rhythmDiagnosis.value = buildRhythmDiagnosis(events, hrv);
  }

  /**
   * Analyze via the backend API. Throws on failure so callers can surface an
   * error and offer retry instead of silently switching to local simulation.
   */
  async function fetchBackendAnalysis(): Promise<void> {
    const response = await fetch(`${backendUrl.value}/ecg/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lead_name: selectedLead.value,
        duration: duration.value,
        sampling_rate: samplingRate.value,
        heart_rate: heartRate.value,
      }),
    });
    if (!response.ok) {
      throw new Error(`分析服务响应异常 (HTTP ${response.status})`);
    }
    const data: ECGAnalysisResponse = await response.json();

    ecgData.value = {
      leadName: data.lead.lead_name,
      samplingRate: data.lead.sampling_rate,
      duration: data.lead.duration,
      samples: data.lead.samples,
      rPeaks: data.lead.r_peaks.map((rp) => ({
        index: rp.index,
        time: rp.time,
        amplitude: rp.amplitude,
      })),
    };
    hrvData.value = {
      heartRate: data.hrv.heart_rate,
      sdnn: data.hrv.sdnn,
      rmssd: data.hrv.rmssd,
      pnn50: data.hrv.pnn50,
      nnIntervals: data.hrv.nn_intervals,
      beatCount: data.hrv.beat_count,
      dataSufficient: data.hrv.data_sufficient,
    };
    arrhythmiaEvents.value = data.arrhythmia_events.map((evt) => ({
      eventType: evt.event_type,
      confidence: evt.confidence,
      description: evt.description,
      timestamp: evt.timestamp,
    }));
    // Use the exact wording produced by the backend; rebuild locally as a
    // fallback only if an older backend omits the field.
    rhythmDiagnosis.value =
      data.rhythm_diagnosis ?? buildRhythmDiagnosis(arrhythmiaEvents.value, hrvData.value);
  }

  /**
   * Run an analysis. Backend mode: show a retryable error on failure and keep
   * the previous results visible; never silently fall back to simulation
   * (that is how mismatched conclusions were produced).
   */
  async function analyzeECG(): Promise<void> {
    isLoading.value = true;
    apiError.value = null;
    try {
      if (useBackend.value) {
        await fetchBackendAnalysis();
      } else {
        runFrontendAnalysis();
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : '未知错误';
      apiError.value = `分析服务暂时不可用：${message}。请检查后端连接后重试。`;
      // eslint-disable-next-line no-console
      console.error('ECG analysis failed:', error);
    } finally {
      isLoading.value = false;
    }
  }

  /** User-initiated retry after a backend failure. */
  async function retryAnalysis(): Promise<void> {
    await analyzeECG();
  }

  /**
   * Start real-time monitoring simulation
   */
  function startMonitoring() {
    isMonitoring.value = true;
    void analyzeECG();
    animationTimer = setInterval(() => {
      // Stop polling automatically while the backend is unavailable: keep the
      // retry banner and last results stable instead of flooding with errors.
      if (useBackend.value && apiError.value) {
        return;
      }
      scrollOffset.value += 5;
      // Regenerate data every full cycle
      if (scrollOffset.value >= currentSamples.value.length) {
        scrollOffset.value = 0;
        void analyzeECG();
      }
    }, 50);
  }

  /**
   * Stop monitoring
   */
  function stopMonitoring() {
    isMonitoring.value = false;
    if (animationTimer) {
      clearInterval(animationTimer);
      animationTimer = null;
    }
  }

  /**
   * Select a different ECG lead
   */
  function selectLead(lead: string) {
    selectedLead.value = lead;
    if (isMonitoring.value) {
      void analyzeECG();
    }
  }

  /**
   * Update heart rate setting
   */
  function setHeartRate(hr: number) {
    heartRate.value = hr;
    if (isMonitoring.value) {
      void analyzeECG();
    }
  }

  return {
    // State
    selectedLead,
    heartRate,
    samplingRate,
    duration,
    isMonitoring,
    ecgData,
    hrvData,
    arrhythmiaEvents,
    rhythmDiagnosis,
    isLoading,
    useBackend,
    backendUrl,
    apiError,
    scrollOffset,
    // Getters
    currentSamples,
    currentRPeaks,
    currentHeartRate,
    dataSufficient,
    // Actions
    analyzeECG,
    retryAnalysis,
    startMonitoring,
    stopMonitoring,
    selectLead,
    setHeartRate,
    generateECGWaveform,
    detectRPeaks,
    calculateHRV,
    detectArrhythmias,
  };
});
