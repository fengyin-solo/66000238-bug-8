export type ArrhythmiaEventType =
  | 'normal'
  | 'insufficient_data'
  | 'tachycardia'
  | 'bradycardia'
  | 'st_elevation'
  | 'atrial_fibrillation'
  | 'premature_ventricular_contraction';

export interface ECGLead {
  leadName: string;
  samplingRate: number;
  duration: number;
  samples: number[];
  rPeaks: RPeak[];
}

export interface RPeak {
  index: number;
  time: number;
  amplitude: number;
}

export interface HRVData {
  /** null when too few beats were detected — display as "数据不足", never 0 */
  heartRate: number | null;
  sdnn: number | null;
  rmssd: number | null;
  pnn50: number | null;
  nnIntervals: number[];
  beatCount: number;
  dataSufficient: boolean;
}

export interface ArrhythmiaEvent {
  eventType: ArrhythmiaEventType;
  confidence: number;
  description: string;
  timestamp: number;
}

/** Backend response uses snake_case; mapped to camelCase in the store. */
export interface ECGAnalysisResponse {
  lead: {
    lead_name: string;
    sampling_rate: number;
    duration: number;
    samples: number[];
    r_peaks: { index: number; time: number; amplitude: number }[];
  };
  hrv: {
    heart_rate: number | null;
    sdnn: number | null;
    rmssd: number | null;
    pnn50: number | null;
    nn_intervals: number[];
    beat_count: number;
    data_sufficient: boolean;
  };
  arrhythmia_events: {
    event_type: ArrhythmiaEventType;
    confidence: number;
    description: string;
    timestamp: number;
  }[];
  rhythm_diagnosis: string;
}

export interface ECGAnalysisRequest {
  leadName: string;
  duration: number;
  samplingRate: number;
  heartRate: number;
}

export const LEAD_NAMES: string[] = [
  'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
];

/** Minimum number of detected beats required for a reliable analysis. */
export const MIN_BEATS_FOR_ANALYSIS = 4;

// Shared wording — must match backend get_rhythm_diagnosis so the status bar,
// result panel and API never present different conclusions for the same data.
export const DIAGNOSIS_INSUFFICIENT_DATA =
  '数据不足 - 有效心跳过少，无法判定心律，请延长记录时间或检查信号质量';
