from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class LeadName(str, Enum):
    I = "I"
    II = "II"
    III = "III"
    aVR = "aVR"
    aVL = "aVL"
    aVF = "aVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"


class ArrhythmiaType(str, Enum):
    NORMAL = "normal"
    INSUFFICIENT_DATA = "insufficient_data"
    TACHYCARDIA = "tachycardia"
    BRADYCARDIA = "bradycardia"
    ST_ELEVATION = "st_elevation"
    AFIB = "atrial_fibrillation"
    PVC = "premature_ventricular_contraction"


class RPeak(BaseModel):
    index: int = Field(..., description="Sample index of R-peak")
    time: float = Field(..., description="Time in seconds")
    amplitude: float = Field(..., description="Amplitude in mV")


class HRVMetrics(BaseModel):
    heart_rate: Optional[float] = Field(
        default=None,
        description="Heart rate in BPM; null when too few beats were detected",
    )
    sdnn: Optional[float] = Field(default=None, description="Standard deviation of NN intervals (ms)")
    rmssd: Optional[float] = Field(
        default=None, description="Root mean square of successive differences (ms)"
    )
    pnn50: Optional[float] = Field(default=None, description="Percentage of successive differences > 50ms")
    nn_intervals: List[float] = Field(default_factory=list, description="NN intervals in ms")
    beat_count: int = Field(default=0, description="Number of detected beats used for the metrics")
    data_sufficient: bool = Field(
        default=False, description="Whether enough beats were detected for a reliable analysis"
    )


class ArrhythmiaEvent(BaseModel):
    event_type: ArrhythmiaType = Field(..., description="Type of arrhythmia detected")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    description: str = Field(..., description="Human-readable description")
    timestamp: float = Field(..., description="Event timestamp in seconds")


class ECGLead(BaseModel):
    lead_name: LeadName = Field(..., description="ECG lead name")
    sampling_rate: int = Field(default=500, description="Sampling rate in Hz")
    duration: float = Field(default=10.0, description="Duration in seconds")
    samples: List[float] = Field(default_factory=list, description="ECG samples in mV")
    r_peaks: List[RPeak] = Field(default_factory=list, description="Detected R-peaks")


class ECGAnalysisRequest(BaseModel):
    lead_name: LeadName = Field(default=LeadName.II, description="ECG lead to analyze")
    duration: float = Field(default=10.0, ge=1.0, le=60.0, description="Duration in seconds")
    sampling_rate: int = Field(default=500, ge=100, le=1000, description="Sampling rate in Hz")
    heart_rate: float = Field(default=72.0, ge=30, le=200, description="Simulated heart rate BPM")


class ECGAnalysisResponse(BaseModel):
    lead: ECGLead = Field(..., description="ECG lead data")
    hrv: HRVMetrics = Field(..., description="HRV analysis results")
    arrhythmia_events: List[ArrhythmiaEvent] = Field(default_factory=list, description="Detected arrhythmia events")
    rhythm_diagnosis: str = Field(..., description="Overall rhythm diagnosis")
