import numpy as np
from scipy import signal
from typing import List, Tuple, Dict, Any, Optional
import re
import math


def gaussian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Generate a Gaussian function for ECG wave simulation."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))


def generate_pqrst_cycle(
    t: np.ndarray,
    heart_rate: float = 72.0,
    lead_config: Dict[str, float] = None,
) -> np.ndarray:
    """
    Generate a single PQRST cycle using Gaussian functions for each wave component.
    
    Each wave (P, Q, R, S, T) is modeled as a Gaussian with specific amplitude,
    center position, and width to simulate realistic ECG morphology.
    """
    if lead_config is None:
        lead_config = {
            "p_amplitude": 0.15,
            "q_amplitude": -0.1,
            "r_amplitude": 1.0,
            "s_amplitude": -0.2,
            "t_amplitude": 0.3,
            "st_elevation": 0.0,
        }

    cycle_duration = 60.0 / heart_rate
    t_normalized = t / cycle_duration

    # P wave: atrial depolarization (starts at ~0ms, peaks at ~80ms)
    p_wave = gaussian(t_normalized, lead_config["p_amplitude"], 0.12, 0.035)

    # Q wave: initial ventricular depolarization (~160ms)
    q_wave = gaussian(t_normalized, lead_config["q_amplitude"], 0.22, 0.012)

    # R wave: main ventricular depolarization (~200ms, tallest peak)
    r_wave = gaussian(t_normalized, lead_config["r_amplitude"], 0.26, 0.012)

    # S wave: late ventricular depolarization (~240ms)
    s_wave = gaussian(t_normalized, lead_config["s_amplitude"], 0.30, 0.015)

    # T wave: ventricular repolarization (~360ms, broader)
    t_wave = gaussian(t_normalized, lead_config["t_amplitude"], 0.48, 0.055)

    # ST segment elevation (if any)
    st_segment = lead_config.get("st_elevation", 0.0) * np.where(
        (t_normalized > 0.32) & (t_normalized < 0.42), 1.0, 0.0
    )

    return p_wave + q_wave + r_wave + s_wave + t_wave + st_segment


def get_lead_config(lead_name: str) -> Dict[str, float]:
    """Get lead-specific configuration for realistic 12-lead ECG simulation."""
    configs = {
        "I": {"p_amplitude": 0.12, "q_amplitude": -0.05, "r_amplitude": 0.8, "s_amplitude": -0.1, "t_amplitude": 0.25, "st_elevation": 0.0},
        "II": {"p_amplitude": 0.15, "q_amplitude": -0.1, "r_amplitude": 1.2, "s_amplitude": -0.2, "t_amplitude": 0.3, "st_elevation": 0.0},
        "III": {"p_amplitude": 0.10, "q_amplitude": -0.08, "r_amplitude": 0.9, "s_amplitude": -0.15, "t_amplitude": 0.2, "st_elevation": 0.0},
        "aVR": {"p_amplitude": -0.10, "q_amplitude": 0.05, "r_amplitude": -0.8, "s_amplitude": 0.1, "t_amplitude": -0.2, "st_elevation": 0.0},
        "aVL": {"p_amplitude": 0.10, "q_amplitude": -0.03, "r_amplitude": 0.6, "s_amplitude": -0.05, "t_amplitude": 0.2, "st_elevation": 0.0},
        "aVF": {"p_amplitude": 0.13, "q_amplitude": -0.09, "r_amplitude": 1.0, "s_amplitude": -0.18, "t_amplitude": 0.28, "st_elevation": 0.0},
        "V1": {"p_amplitude": 0.08, "q_amplitude": 0.0, "r_amplitude": 0.3, "s_amplitude": -0.8, "t_amplitude": 0.15, "st_elevation": 0.0},
        "V2": {"p_amplitude": 0.10, "q_amplitude": -0.02, "r_amplitude": 0.6, "s_amplitude": -0.6, "t_amplitude": 0.25, "st_elevation": 0.0},
        "V3": {"p_amplitude": 0.10, "q_amplitude": -0.05, "r_amplitude": 0.9, "s_amplitude": -0.4, "t_amplitude": 0.3, "st_elevation": 0.0},
        "V4": {"p_amplitude": 0.12, "q_amplitude": -0.08, "r_amplitude": 1.3, "s_amplitude": -0.25, "t_amplitude": 0.35, "st_elevation": 0.0},
        "V5": {"p_amplitude": 0.12, "q_amplitude": -0.1, "r_amplitude": 1.1, "s_amplitude": -0.15, "t_amplitude": 0.3, "st_elevation": 0.0},
        "V6": {"p_amplitude": 0.10, "q_amplitude": -0.08, "r_amplitude": 0.9, "s_amplitude": -0.1, "t_amplitude": 0.25, "st_elevation": 0.0},
    }
    return configs.get(lead_name, configs["II"])


def generate_ecg_signal(
    lead_name: str = "II",
    duration: float = 10.0,
    sampling_rate: int = 500,
    heart_rate: float = 72.0,
    noise_level: float = 0.02,
    include_arrhythmia: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a realistic ECG signal for a specified lead.
    
    Args:
        lead_name: ECG lead name (I, II, III, aVR, aVL, aVF, V1-V6)
        duration: Signal duration in seconds
        sampling_rate: Sampling rate in Hz
        heart_rate: Heart rate in BPM
        noise_level: Baseline noise amplitude
        include_arrhythmia: Whether to simulate arrhythmia events
        
    Returns:
        Tuple of (time_array, ecg_signal)
    """
    total_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, total_samples)
    ecg = np.zeros(total_samples)

    lead_config = get_lead_config(lead_name)
    cycle_duration = 60.0 / heart_rate

    # Generate consecutive PQRST cycles
    beat_count = 0
    for start_time in np.arange(0, duration, cycle_duration):
        end_time = min(start_time + cycle_duration, duration)
        mask = (t >= start_time) & (t < end_time)
        if not np.any(mask):
            continue

        t_cycle = t[mask] - start_time
        
        # Add slight HRV variation to each beat
        hrv_factor = 1.0 + np.random.normal(0, 0.02)
        modified_hr = heart_rate * hrv_factor
        cycle_config = lead_config.copy()

        # Simulate arrhythmia if requested
        if include_arrhythmia and beat_count > 2:
            if np.random.random() < 0.1:  # 10% chance of PVC
                cycle_config["r_amplitude"] *= 1.8
                cycle_config["t_amplitude"] *= -0.5
                cycle_config["q_amplitude"] *= 0.5

        ecg[mask] += generate_pqrst_cycle(t_cycle, modified_hr, cycle_config)
        beat_count += 1

    # Add baseline wander (low-frequency noise ~0.15 Hz)
    baseline_wander = 0.03 * np.sin(2 * np.pi * 0.15 * t)
    
    # Add high-frequency noise (muscle artifact)
    noise = noise_level * np.random.randn(total_samples)

    ecg = ecg + baseline_wander + noise

    return t, ecg


def pan_tompkins_r_peak_detection(
    ecg_signal: np.ndarray, sampling_rate: int = 500
) -> List[Dict[str, Any]]:
    """
    Simplified Pan-Tompkins algorithm for R-peak detection.
    
    Steps:
    1. Bandpass filter (5-15 Hz)
    2. Differentiation
    3. Squaring
    4. Moving window integration
    5. Adaptive thresholding for peak detection
    """
    # Step 1: Bandpass filter (5-15 Hz) to isolate QRS complex
    nyquist = sampling_rate / 2
    low = 5.0 / nyquist
    high = 15.0 / nyquist
    b, a = signal.butter(2, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, ecg_signal)

    # Step 2: Differentiation - highlights QRS slopes
    diff_signal = np.diff(filtered)

    # Step 3: Squaring - emphasizes large differences
    squared = diff_signal ** 2

    # Step 4: Moving window integration (150ms window)
    window_size = int(0.15 * sampling_rate)
    kernel = np.ones(window_size) / window_size
    integrated = np.convolve(squared, kernel, mode="same")

    # Step 5: Adaptive threshold peak detection
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    min_distance = int(0.2 * sampling_rate)  # Minimum 200ms between peaks

    r_peaks = []
    above_threshold = integrated > threshold
    last_peak = -min_distance

    for i in range(1, len(integrated) - 1):
        if above_threshold[i] and i - last_peak >= min_distance:
            # Find the actual peak in the original signal within a window
            search_start = max(0, i - window_size // 2)
            search_end = min(len(ecg_signal), i + window_size // 2)
            local_peak = search_start + np.argmax(ecg_signal[search_start:search_end])

            if local_peak not in [rp["index"] for rp in r_peaks]:
                r_peaks.append({
                    "index": int(local_peak),
                    "time": float(local_peak / sampling_rate),
                    "amplitude": float(ecg_signal[local_peak]),
                })
                last_peak = i

    return r_peaks


def calculate_hrv(r_peaks: List[Dict[str, Any]], sampling_rate: int = 500) -> Dict[str, Any]:
    """
    Calculate Heart Rate Variability (HRV) metrics from R-peak positions.
    
    Metrics:
    - Heart Rate (BPM)
    - SDNN: Standard deviation of NN intervals
    - RMSSD: Root mean square of successive differences
    - pNN50: Percentage of successive differences > 50ms
    """
    if len(r_peaks) < 3:
        return {
            "heart_rate": 0.0,
            "sdnn": 0.0,
            "rmssd": 0.0,
            "pnn50": 0.0,
            "nn_intervals": [],
        }

    # Calculate RR intervals in milliseconds
    rr_intervals = []
    for i in range(1, len(r_peaks)):
        rr = (r_peaks[i]["index"] - r_peaks[i - 1]["index"]) / sampling_rate * 1000
        rr_intervals.append(rr)

    rr_array = np.array(rr_intervals)

    # Heart rate from mean RR interval
    mean_rr = np.mean(rr_array)
    heart_rate = 60000.0 / mean_rr if mean_rr > 0 else 0.0

    # SDNN: Standard deviation of all NN intervals
    sdnn = float(np.std(rr_array))

    # RMSSD: Root mean square of successive differences
    successive_diffs = np.diff(rr_array)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0

    # pNN50: Percentage of successive differences > 50ms
    if len(successive_diffs) > 0:
        nn50_count = np.sum(np.abs(successive_diffs) > 50)
        pnn50 = float(nn50_count / len(successive_diffs) * 100)
    else:
        pnn50 = 0.0

    return {
        "heart_rate": round(heart_rate, 1),
        "sdnn": round(sdnn, 2),
        "rmssd": round(rmssd, 2),
        "pnn50": round(pnn50, 2),
        "nn_intervals": [round(float(x), 2) for x in rr_intervals],
    }


def detect_arrhythmia(
    r_peaks: List[Dict[str, Any]],
    hrv: Dict[str, Any],
    ecg_signal: np.ndarray,
    sampling_rate: int = 500,
) -> List[Dict[str, Any]]:
    """
    Detect arrhythmia events based on R-peaks, HRV metrics, and signal morphology.

    Detects:
    - Insufficient data: fewer than 3 detectable beats (HR not computable)
    - Tachycardia: HR > 100 BPM
    - Bradycardia: HR < 60 BPM
    - ST-segment elevation: potential myocardial infarction
    - Irregular rhythm patterns

    心跳不足或心率算不出来时只输出 insufficient_data，
    不允许判成 normal，也不允许判成任何心律失常。
    """
    heart_rate = hrv["heart_rate"]

    # Insufficient data: not enough beats to estimate a rhythm
    if len(r_peaks) < 3 or heart_rate <= 0:
        return [
            {
                "event_type": "insufficient_data",
                "confidence": 1.0,
                "description": (
                    f"有效心跳不足（检测到 {len(r_peaks)} 次，至少需要 3 次），数据不足以判读心律"
                ),
                "timestamp": r_peaks[0]["time"] if r_peaks else 0.0,
            }
        ]

    events = []

    # Tachycardia detection
    if heart_rate > 100:
        events.append({
            "event_type": "tachycardia",
            "confidence": min(1.0, (heart_rate - 100) / 50 + 0.6),
            "description": f"心率过快 ({heart_rate:.0f} BPM)，检测到心动过速",
            "timestamp": r_peaks[0]["time"],
        })

    # Bradycardia detection (heart_rate <= 0 has already been handled above)
    if heart_rate < 60:
        events.append({
            "event_type": "bradycardia",
            "confidence": min(1.0, (60 - heart_rate) / 30 + 0.6),
            "description": f"心率过慢 ({heart_rate:.0f} BPM)，检测到心动过缓",
            "timestamp": r_peaks[0]["time"],
        })

    # ST-segment elevation detection
    st_event = _detect_st_elevation(r_peaks, hrv, ecg_signal, sampling_rate)
    if st_event is not None:
        events.append(st_event)

    # Irregular rhythm detection (high SDNN relative to mean)
    if len(hrv.get("nn_intervals", [])) > 3:
        nn_array = np.array(hrv["nn_intervals"])
        mean_nn = np.mean(nn_array)
        cv = np.std(nn_array) / mean_nn if mean_nn > 0 else 0
        if cv > 0.15:
            events.append({
                "event_type": "atrial_fibrillation",
                "confidence": min(1.0, cv * 2),
                "description": "RR 间期不规则，可能提示房颤",
                "timestamp": r_peaks[0]["time"],
            })

    # Normal rhythm
    if not events:
        events.append({
            "event_type": "normal",
            "confidence": 1.0,
            "description": "正常窦性心律",
            "timestamp": r_peaks[0]["time"],
        })

    return events


# ST 抬高判定阈值 (mV)：ST 段相对等电位基线的抬升幅度
ST_ELEVATION_THRESHOLD = 0.1
# 至少需要多少个可测量的心拍才允许判读 ST 段
ST_MIN_VALID_BEATS = 3


def _beat_baseline(
    ecg_signal: np.ndarray, idx: int, rr_samples: float
) -> Optional[float]:
    """
    Estimate the isoelectric baseline for a beat from the TP segment.

    Uses a window 0.30~0.24 RR before the R-peak (the flat interval after the
    previous beat's T wave); window width is 0.06 RR so enough samples remain
    even at high heart rates. Returns None when the signal is too short.
    """
    width = max(4, int(0.06 * rr_samples))
    end = idx - int(0.24 * rr_samples)
    start = max(0, end - width)
    if end <= 0 or end - start < 3:
        return None
    return float(np.mean(ecg_signal[start:end]))


def _measure_st_elevation(
    ecg_signal: np.ndarray, idx: int, rr_samples: float
) -> Optional[float]:
    """
    Measure one beat's ST-segment elevation (mV) above baseline.

    The ST window spans 0.06~0.10 RR after the R-peak (around J+60ms, scaled
    with beat length). A fixed millisecond window lands on the T wave at high
    heart rates and produces false positives; T-wave upstroke only starts at
    ~0.14 RR, so this window stays on the actual ST segment for any rhythm.
    Returns None for beats that cannot be measured (truncated/insufficient).
    """
    st_start = idx + int(0.06 * rr_samples)
    st_end = idx + int(0.10 * rr_samples)
    if st_end >= len(ecg_signal):
        return None
    baseline = _beat_baseline(ecg_signal, idx, rr_samples)
    if baseline is None:
        return None
    return float(np.mean(ecg_signal[st_start:st_end]) - baseline)


def _detect_st_elevation(
    r_peaks: List[Dict[str, Any]],
    hrv: Dict[str, Any],
    ecg_signal: np.ndarray,
    sampling_rate: int,
) -> Optional[Dict[str, Any]]:
    """
    Per-beat ST measurement with RR-adaptive windows. An ST elevation event is
    reported only when more than half of the *measurable* beats (and at least
    ST_MIN_VALID_BEATS) exceed the threshold. Unmeasurable beats are skipped
    instead of being counted as elevated.
    """
    rr_samples_list = [rr / 1000.0 * sampling_rate for rr in hrv.get("nn_intervals", [])]
    if not rr_samples_list:
        return None

    elevated = 0
    measured = 0
    for i, rp in enumerate(r_peaks):
        # Last beat has no following RR; reuse the previous interval.
        rr_samples = rr_samples_list[min(i, len(rr_samples_list) - 1)]
        elevation = _measure_st_elevation(ecg_signal, rp["index"], rr_samples)
        if elevation is None:
            continue
        measured += 1
        if elevation > ST_ELEVATION_THRESHOLD:
            elevated += 1

    if measured >= ST_MIN_VALID_BEATS and elevated > measured * 0.5:
        return {
            "event_type": "st_elevation",
            "confidence": min(1.0, elevated / measured),
            "description": "检测到 ST 段抬高，可能提示心肌梗死",
            "timestamp": r_peaks[0]["time"],
        }
    return None


def get_rhythm_diagnosis(arrhythmia_events: List[Dict[str, Any]], hrv: Dict[str, Any]) -> str:
    """Generate overall rhythm diagnosis based on detected events and HRV."""
    event_types = [e["event_type"] for e in arrhythmia_events]

    if "insufficient_data" in event_types:
        beat_count = _extract_beat_count(arrhythmia_events)
        return (
            f"数据不足，无法判读心律 | 有效心跳: {beat_count} 次 | 心率无法计算"
        )
    elif "st_elevation" in event_types:
        return "ST 段抬高 - 建议立即就医检查"
    elif "tachycardia" in event_types and "atrial_fibrillation" in event_types:
        return "快速房颤 - 建议进一步心脏评估"
    elif "tachycardia" in event_types:
        return "窦性心动过速 - 请结合临床症状判断"
    elif "bradycardia" in event_types:
        return "窦性心动过缓 - 建议关注心率变化"
    elif "atrial_fibrillation" in event_types:
        return "心律不规则 - 疑似房颤，建议 Holter 监测"
    else:
        hr = hrv.get("heart_rate", 0)
        sdnn = hrv.get("sdnn", 0)
        return f"正常窦性心律 | HR: {hr:.0f} BPM | SDNN: {sdnn:.1f} ms"


def _extract_beat_count(events: List[Dict[str, Any]]) -> int:
    for event in events:
        if event.get("event_type") != "insufficient_data":
            continue
        match = re.search(r"检测到\s*(\d+)\s*次", event.get("description", ""))
        if match:
            return int(match.group(1))
    return 0
