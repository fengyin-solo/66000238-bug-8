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
  heartRate: number;
  sdnn: number;
  rmssd: number;
  pnn50: number;
  nnIntervals: number[];
}

export type ArrhythmiaEventType =
  | 'normal'
  | 'tachycardia'
  | 'bradycardia'
  | 'st_elevation'
  | 'atrial_fibrillation'
  | 'premature_ventricular_contraction'
  | 'insufficient_data';

export interface ArrhythmiaEvent {
  eventType: ArrhythmiaEventType;
  confidence: number;
  description: string;
  timestamp: number;
}

export interface RPeakDTO {
  index: number;
  time: number;
  amplitude: number;
}

export interface ECGLeadDTO {
  lead_name: string;
  sampling_rate: number;
  duration: number;
  samples: number[];
  r_peaks: RPeakDTO[];
}

export interface HRVDataDTO {
  heart_rate: number;
  sdnn: number;
  rmssd: number;
  pnn50: number;
  nn_intervals: number[];
}

export interface ArrhythmiaEventDTO {
  event_type: ArrhythmiaEventType;
  confidence: number;
  description: string;
  timestamp: number;
}

export interface ECGAnalysisResponse {
  lead: ECGLeadDTO;
  hrv: HRVDataDTO;
  arrhythmia_events: ArrhythmiaEventDTO[];
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
