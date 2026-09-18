import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import type { ECGLead, HRVData, ArrhythmiaEvent, ECGAnalysisResponse } from '../types';
import {
  detectRPeaks,
  calculateHRV,
  detectArrhythmias,
  buildRhythmDiagnosis,
} from '../utils/analysis';

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
  // 后端接口不可用时的错误信息；为空表示无错误。失败时保留上一次分析结果，
  // 由界面提示用户重试，而不是静默回退到本地模拟（那会让两处口径看似不一致）。
  const backendError = ref<string>('');

  let animationTimer: ReturnType<typeof setInterval> | null = null;
  let scrollOffset = ref<number>(0);

  // Getters
  const currentSamples = computed(() => ecgData.value?.samples ?? []);
  const currentRPeaks = computed(() => ecgData.value?.rPeaks ?? []);
  // 只反映本次分析实测心率；算不出来就是 0，由界面按“数据不足”展示
  const currentHeartRate = computed(() => hrvData.value?.heartRate ?? 0);

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
   * Run full ECG analysis (frontend simulation)
   */
  async function analyzeECG() {
    isLoading.value = true;
    backendError.value = '';

    if (useBackend.value) {
      try {
        await runBackendAnalysis();
      } catch (error) {
        // 接口暂时不可用：明确提示并保留重试入口，不静默用本地数据冒充
        backendError.value =
          error instanceof Error ? error.message : '后端分析服务暂时不可用，请稍后重试';
        // 监测循环中失败则停止自动轮询，避免反复刷错误
        stopMonitoring();
      } finally {
        isLoading.value = false;
      }
      return;
    }

    runFrontendAnalysis();
    isLoading.value = false;
  }

  async function runBackendAnalysis() {
    let response: Response;
    try {
      response = await fetch(`${backendUrl.value}/ecg/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lead_name: selectedLead.value,
          duration: duration.value,
          sampling_rate: samplingRate.value,
          heart_rate: heartRate.value,
        }),
      });
    } catch {
      throw new Error(
        `无法连接分析服务 (${backendUrl.value})，请确认后端已启动后点击“重试”`,
      );
    }

    if (!response.ok) {
      throw new Error(`分析服务暂不可用 (HTTP ${response.status})，请稍后点击“重试”`);
    }

    const data: ECGAnalysisResponse = await response.json();
    applyAnalysisResult(
      {
        leadName: data.lead.lead_name,
        samplingRate: data.lead.sampling_rate,
        duration: data.lead.duration,
        samples: data.lead.samples,
        rPeaks: data.lead.r_peaks.map((rp) => ({
          index: rp.index,
          time: rp.time,
          amplitude: rp.amplitude,
        })),
      },
      {
        heartRate: data.hrv.heart_rate,
        sdnn: data.hrv.sdnn,
        rmssd: data.hrv.rmssd,
        pnn50: data.hrv.pnn50,
        nnIntervals: data.hrv.nn_intervals,
      },
      data.arrhythmia_events.map((evt) => ({
        eventType: evt.event_type,
        confidence: evt.confidence,
        description: evt.description,
        timestamp: evt.timestamp,
      })),
      data.rhythm_diagnosis,
    );
  }

  function runFrontendAnalysis() {
    const lead = generateECGWaveform();
    const peaks = detectRPeaks(lead.samples, lead.samplingRate);
    lead.rPeaks = peaks;

    const hrv = calculateHRV(peaks, lead.samplingRate);
    const events = detectArrhythmias(hrv, peaks, lead.samples, lead.samplingRate);

    applyAnalysisResult(lead, hrv, events, buildRhythmDiagnosis(events, hrv));
  }

  /**
   * 统一写入分析结果。诊断文案与事件列表同源：
   * 后端结果直接采用后端文案，本地结果由与后端一致的 builder 生成，
   * 保证分析页面状态栏与结果面板口径一致。
   */
  function applyAnalysisResult(
    lead: ECGLead,
    hrv: HRVData,
    events: ArrhythmiaEvent[],
    diagnosis: string,
  ) {
    ecgData.value = lead;
    hrvData.value = hrv;
    arrhythmiaEvents.value = events;
    rhythmDiagnosis.value = diagnosis;
  }

  /**
   * Start real-time monitoring simulation
   */
  function startMonitoring() {
    isMonitoring.value = true;
    analyzeECG();
    animationTimer = setInterval(() => {
      scrollOffset.value += 5;
      // Regenerate data every full cycle
      if (scrollOffset.value >= currentSamples.value.length) {
        scrollOffset.value = 0;
        analyzeECG();
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
      analyzeECG();
    }
  }

  /**
   * Update heart rate setting
   */
  function setHeartRate(hr: number) {
    heartRate.value = hr;
    if (isMonitoring.value) {
      analyzeECG();
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
    backendError,
    scrollOffset,
    // Getters
    currentSamples,
    currentRPeaks,
    currentHeartRate,
    // Actions
    analyzeECG,
    startMonitoring,
    stopMonitoring,
    selectLead,
    setHeartRate,
    generateECGWaveform,
    detectRPeaks,
  };
});
