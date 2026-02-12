from flask import Flask, render_template, jsonify, request, redirect, url_for
import sys
import os
from pathlib import Path

# --- FOLDER RESTRUCTURE FIX ---
# Add project root and sibling folders to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "database"))
sys.path.append(str(BASE_DIR / "xai"))
sys.path.append(str(BASE_DIR / "models_training"))

import db_service
import json
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import resample_poly, butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
from werkzeug.utils import secure_filename
from typing import List, Dict, Any, Tuple
import warnings
import psycopg2
import subprocess

# XAI – Option A (clinical text + model prediction)
from xai import explain_segment, explain_decision, reset_model
from decision_engine.rhythm_orchestrator import RhythmOrchestrator
from decision_engine.models import SegmentDecision
from data_loader import CLASS_NAMES, RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES

# Suppress harmless scipy warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# Flask App & Global Config
# =========================================================

app = Flask(__name__)

TARGET_FS = 250
SEGMENT_DURATION_S = 10.0
SEGMENT_LENGTH = int(TARGET_FS * SEGMENT_DURATION_S)
HRV_INTERP_FS = 4.0

# Where individual uploaded JSONs go
# Point to shared 'data/ecg_data' folder
DATA_ROOT = BASE_DIR / "data"
app.config["UPLOAD_FOLDER"] = str(DATA_ROOT / "ecg_data")
DATA_ROOT_DIR = Path(app.config["UPLOAD_FOLDER"])
os.makedirs(DATA_ROOT_DIR, exist_ok=True)

# Folder that holds the bulk JSON datasets already converted
DATASET_JSON_DIR = DATA_ROOT / "input_segments"
os.makedirs(DATASET_JSON_DIR, exist_ok=True)


# =========================================================
# ECG Loading & Preprocessing
# =========================================================

def _load_data_from_json(file_path: Path) -> Tuple[np.ndarray, int]:
    """
    Loads ECG data from a JSON file and returns (signal, original_fs).
    Supports:
      - SensorData[0]["ECG_CH_A"]
      - Top-level "ECG_CH_A" / "ECG_CH_B"
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    signal = None
    original_fs = 250
    filename = file_path.name

    # Structure 1: SensorData list (PTB-XL style, etc.)
    if isinstance(data.get("SensorData"), list) and data["SensorData"]:
        row = data["SensorData"][0]
        if "ECG_CH_A" in row:
            signal = np.array(row["ECG_CH_A"], dtype=float)

        # Heuristics for fs
        if "PTBXL" in filename.upper() or "PTB-XL" in filename.upper():
            original_fs = 500  # many PTB-XL records
        elif "MITDB" in filename.upper() or "MIT-BIH" in filename.upper():
            original_fs = 360

    # Structure 2: top-level "ECG_CH_A"
    elif "ECG_CH_A" in data:
        signal = np.array(data["ECG_CH_A"], dtype=float)
        if "MITDB" in filename.upper() or "MIT-BIH" in filename.upper():
            original_fs = 360

    elif "ECG_CH_B" in data:
        signal = np.array(data["ECG_CH_B"], dtype=float)

    if signal is None:
        raise ValueError(f"Could not find valid ECG channel in JSON file: {filename}")

    return signal, original_fs


from signal_processing.cleaning import clean_signal

def _preprocess(signal: np.ndarray, original_fs: int) -> np.ndarray:
    """
    Standard preprocessing using signal_processing module.
    1. Resample to Target Rate
    2. Clean (Baseline Removal + Powerline Removal)
    """
    # Resample first if needed
    if original_fs != TARGET_FS:
        signal = resample_poly(signal, TARGET_FS, original_fs).astype(np.float32)
    else:
        signal = signal.astype(np.float32)
        
    # Apply centralized cleaning
    return clean_signal(signal, TARGET_FS)


def _detect_r_peaks_neurokit(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Robust R-peak detection using NeuroKit2.
    """
    try:
        # 1. Clean signal (removes baseline wander, powerline noise)
        cleaned = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
        # 2. Find Peaks
        # method='neurokit' is steep-slope based, very good for QRS
        signals, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
        peaks = info.get("ECG_R_Peaks", [])
        # Ensure we return int array
        return np.array(peaks, dtype=int)
    except Exception as e:
        print(f"NeuroKit Peak Detection Failed: {e}")
        return np.array([], dtype=int)

def _r_peak_detection(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Wrapper for NeuroKit detection to maintain compatibility.
    """
    return _detect_r_peaks_neurokit(signal, fs)


# =========================================================
# HRV, Morphology & PR/QRS Features
# =========================================================

def _calculate_frequency_hrv(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
    """
    Frequency-domain HRV (VLF, LF, HF, LF/HF) using Welch.
    """
    out = {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}
    if len(rr_intervals_ms) < 5:
        return out

    rr_s = rr_intervals_ms / 1000.0
    t = np.cumsum(rr_s)
    t -= t[0]

    try:
        f_interp = interp1d(t, rr_intervals_ms, kind="cubic")
        t_new = np.arange(t[0], t[-1], 1.0 / HRV_INTERP_FS)
        rr_interp = f_interp(t_new)
    except ValueError:
        return out

    n = len(rr_interp)
    if n < 16:
        return out

    nperseg = min(n, 256)
    noverlap = nperseg // 2

    fxx, pxx = welch(
        rr_interp,
        fs=HRV_INTERP_FS,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    def band_power(f, p, band):
        idx = (f >= band[0]) & (f < band[1])
        if not np.any(idx):
            return 0.0
        return float(np.trapz(p[idx], f[idx]))

    vlf = band_power(fxx, pxx, (0.0, 0.04))
    lf = band_power(fxx, pxx, (0.04, 0.15))
    hf = band_power(fxx, pxx, (0.15, 0.4))

    out["VLF"] = vlf
    out["LF"] = lf
    out["HF"] = hf
    out["LF_HF_ratio"] = float(lf / hf) if lf > 0 and hf > 0 else 0.0
    return out


def _calculate_nonlinear_hrv(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
    """
    Poincaré SD1/SD2.
    """
    out = {"SD1": 0.0, "SD2": 0.0}
    if len(rr_intervals_ms) < 2:
        return out

    rr_n = rr_intervals_ms[:-1]
    rr_n1 = rr_intervals_ms[1:]

    sd1 = np.sqrt(0.5 * np.var(rr_n - rr_n1))
    sd2 = np.sqrt(2 * np.var(rr_intervals_ms) - sd1**2)

    out["SD1"] = float(sd1)
    out["SD2"] = float(sd2)
    return out


def _compute_qrs_durations(segment: np.ndarray, segment_r_peaks: np.ndarray, fs: int) -> np.ndarray:
    """
    Estimate QRS durations using NeuroKit2 Delineation.
    Returns array of durations in ms for each QRS detected.
    """
    if segment_r_peaks is None or len(segment_r_peaks) == 0:
        return np.array([])
    
    try:
        # NeuroKit DWT (Discrete Wavelet Transform) is robust for delineation
        # It needs R-peaks. We pass current R-peaks to help it.
        # Note: ecg_delineate DWT method is fast and accurate.
        # But DWT sometimes ignores our R-peaks and finds its own if not aligned? 
        # Actually it uses R-peaks to find QRS onset/offset around them.
        
        _, waves = nk.ecg_delineate(segment, segment_r_peaks, sampling_rate=fs, method="dwt", show=False)
        
        # waves dictionary contains "ECG_R_Onsets" and "ECG_R_Offsets"
        # These are lists with NaNs for missing waves
        r_onsets = np.array(waves.get("ECG_R_Onsets", []))
        r_offsets = np.array(waves.get("ECG_R_Offsets", []))
        
        # Ensure we have data
        if len(r_onsets) == 0 or len(r_offsets) == 0:
            return np.array([])
            
        # Create mask for valid pairs
        valid_mask = ~pd.isna(r_onsets) & ~pd.isna(r_offsets)
        
        if np.sum(valid_mask) == 0:
            return np.array([])
            
        # Calculate Durations
        durations_ms = (r_offsets[valid_mask] - r_onsets[valid_mask]) * 1000.0 / fs
        
        # Filter physiological range (e.g. 40ms to 250ms)
        durations_ms = durations_ms[(durations_ms >= 30) & (durations_ms <= 300)]
        
        return durations_ms

    except Exception as e:
        print(f"NeuroKit QRS Calc Failed: {e}")
        return np.array([80.0]) # Fallback default



def _calculate_morphology_features(segment: np.ndarray, segment_r_peaks: np.ndarray) -> Dict[str, Any]:
    """
    QRS energy + QRS duration distribution (ms).
    """
    out: Dict[str, Any] = {
        "QRS_Avg_Energy": 0.0,
        "QRS_Energy_Std": 0.0,
        "qrs_durations_ms": [],
    }
    if len(segment_r_peaks) == 0:
        return out

    window_samples = int(0.100 * TARGET_FS)
    energies = []

    for r in segment_r_peaks:
        start = max(0, r - window_samples)
        end = min(len(segment), r + window_samples)
        qrs_seg = segment[start:end]
        energies.append(np.sum(qrs_seg**2))

    if energies:
        out["QRS_Avg_Energy"] = float(np.mean(energies))
        out["QRS_Energy_Std"] = float(np.std(energies))

    qrs_list = _compute_qrs_durations(segment, segment_r_peaks, TARGET_FS)
    out["qrs_durations_ms"] = qrs_list.tolist() if qrs_list.size > 0 else []
    return out


def _sanitize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure no NaN/Inf for JSONB."""
    clean = {}
    for k, v in features.items():
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                clean[k] = 0.0
            else:
                clean[k] = float(v)
        else:
            clean[k] = v
    return clean


def _calculate_pr_interval(signal: np.ndarray, r_peaks: np.ndarray, fs: int) -> float:
    """
    Estimate PR interval using NeuroKit2 Delineation.
    Returns median PR interval in ms.
    """
    if r_peaks is None or len(r_peaks) == 0:
        return 0.0

    try:
        # Use NeuroKit's DWT method for delineation
        _, waves = nk.ecg_delineate(signal, r_peaks, sampling_rate=fs, method="dwt", show=False)
        
        # P-onset to R-onset (or Q-wave start)
        p_onsets = np.array(waves.get("ECG_P_Onsets", []))
        r_onsets = np.array(waves.get("ECG_R_Onsets", []))
        
        if len(p_onsets) == 0 or len(r_onsets) == 0:
            return 0.0
            
        valid = ~pd.isna(p_onsets) & ~pd.isna(r_onsets)
        
        if np.sum(valid) == 0:
            return 0.0
            
        pr_vals = (r_onsets[valid] - p_onsets[valid]) * 1000.0 / fs
        
        # Filter valid range (80 - 450 ms)
        pr_vals = pr_vals[(pr_vals >= 80) & (pr_vals <= 450)]
        
        if len(pr_vals) == 0:
             return 0.0
             
        return float(np.nanmedian(pr_vals))
        
    except Exception as e:
        print(f"NK PR Failed: {e}")
        return 0.0


def _extract_segment_features(
    segment: np.ndarray, segment_r_peaks: np.ndarray, segment_idx: int
) -> Dict[str, Any]:
    """
    Full time-domain, HRV, and morphology features per 10 s segment.
    """
    features: Dict[str, Any] = {}

    features["segment_index"] = int(segment_idx)
    features["mean_amplitude"] = float(np.mean(segment))
    features["std_amplitude"] = float(np.std(segment))

    rr_intervals_ms = np.array([])
    if len(segment_r_peaks) >= 2:
        rr_samples = np.diff(segment_r_peaks)
        rr_intervals_ms = rr_samples * 1000.0 / TARGET_FS

    if rr_intervals_ms.size > 0:
        features["rr_intervals_ms"] = rr_intervals_ms.tolist()
        features["mean_rr"] = float(np.mean(rr_intervals_ms))
        features["mean_hr"] = (
            float(60.0 / (features["mean_rr"] / 1000.0))
            if features["mean_rr"] > 0
            else 0.0
        )
        features["SDNN"] = float(np.std(rr_intervals_ms))
        diff_rr = np.diff(rr_intervals_ms)
        features["RMSSD"] = (
            float(np.sqrt(np.mean(diff_rr**2))) if diff_rr.size > 0 else 0.0
        )
        features["pNN50"] = (
            float(np.sum(np.abs(diff_rr) > 50) / diff_rr.size)
            if diff_rr.size > 0
            else 0.0
        )
    else:
        features["rr_intervals_ms"] = []
        features["mean_rr"] = 0.0
        features["mean_hr"] = 0.0
        features["SDNN"] = 0.0
        features["RMSSD"] = 0.0
        features["pNN50"] = 0.0

    features.update(_calculate_morphology_features(segment, segment_r_peaks))
    features.update(_calculate_frequency_hrv(rr_intervals_ms))
    features.update(_calculate_nonlinear_hrv(rr_intervals_ms))

    # Calculate PR interval from waveform
    pr_interval = _calculate_pr_interval(segment, segment_r_peaks, TARGET_FS)
    features["pr_interval"] = float(pr_interval)
    
    return features


# =========================================================
# Ingestion: process an uploaded file & save features to SQL
# =========================================================

def process_and_save_record(file_path: Path) -> str:
    """
    Process one uploaded ECG JSON file:
      - preprocess
      - R-peaks
      - segment into 10s
      - compute features
      - store in ecg_features_annotatable
    """
    filename_key = str(file_path.relative_to(DATA_ROOT_DIR))

    try:
        raw_signal, original_fs = _load_data_from_json(file_path)
        processed_signal = _preprocess(raw_signal, original_fs)
        r_peaks_all = _r_peak_detection(processed_signal, TARGET_FS)
    except Exception as e:
        raise Exception(f"Processing failed for {filename_key}: {e}")

    conn = None
    try:
        conn = db_service._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ecg_features_annotatable (
                    segment_id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    segment_index INT NOT NULL,
                    segment_start_s FLOAT NOT NULL,
                    segment_duration_s FLOAT NOT NULL,
                    arrhythmia_label VARCHAR(50) DEFAULT NULL,
                    arrhythmia_text_notes TEXT DEFAULT '',
                    r_peaks_in_segment TEXT,
                    features_json JSONB,
                    model_pred_label TEXT,
                    model_pred_probs JSONB,
                    cardiologist_notes TEXT,
                    corrected_by TEXT,
                    corrected_at TIMESTAMP,
                    training_round INT
                );
                
                CREATE TABLE IF NOT EXISTS ecg_segments (
                    segment_id SERIAL PRIMARY KEY,
                    patient_id TEXT,
                    filename VARCHAR(255) NOT NULL,
                    segment_index INT NOT NULL,
                    signal JSONB,
                    features JSONB,
                    segment_state TEXT,
                    background_rhythm TEXT,
                    events_json JSONB,
                    segment_fs INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_segment
                    ON ecg_features_annotatable (filename, segment_index);
                
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_ecg_segments
                    ON ecg_segments (filename, segment_index);
                """
            )
        conn.commit()

        n_segments = len(processed_signal) // SEGMENT_LENGTH

        with conn.cursor() as cur:
            for i in range(n_segments):
                start = i * SEGMENT_LENGTH
                end = (i + 1) * SEGMENT_LENGTH
                segment = processed_signal[start:end]

                seg_r_peaks_abs = r_peaks_all[
                    (r_peaks_all >= start) & (r_peaks_all < end)
                ]
                seg_r_peaks_rel = seg_r_peaks_abs - start

                feats = _extract_segment_features(segment, seg_r_peaks_rel, i)
                feats_clean = _sanitize_features(feats)
                rpeaks_str = ",".join(map(str, seg_r_peaks_rel))
                segment_start_s = start / TARGET_FS

                cur.execute(
                    """
                    INSERT INTO ecg_features_annotatable
                    (filename, segment_index, segment_start_s, segment_duration_s,
                     r_peaks_in_segment, features_json)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (filename, segment_index) DO NOTHING
                    """,
                    (
                        filename_key,
                        i,
                        segment_start_s,
                        SEGMENT_DURATION_S,
                        rpeaks_str,
                        json.dumps(feats_clean),
                    ),
                )

                # DUAL INSERT: Patch the Ingestion Gap by writing to the new table too
                cur.execute(
                    """
                    INSERT INTO ecg_segments
                    (filename, segment_index, signal, features, segment_state, segment_fs, events_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (filename, segment_index) DO NOTHING
                    """,
                    (
                        filename_key,
                        i,
                        json.dumps(segment.tolist()),
                        json.dumps(feats_clean),
                        'ANALYZED',
                        TARGET_FS,
                        '[]'
                    ),
                )
        conn.commit()
        return filename_key

    except psycopg2.Error as e:
        raise Exception(f"Database error during insert: {e}")
    finally:
        if conn:
            conn.close()


# =========================================================
# Utility: load a segment signal from disk for plotting/XAI
# =========================================================

def _load_and_segment_raw_data(relative_path: str, segment_index: int) -> List[float]:
    """
    Load raw ECG segment used for plotting and XAI.

    1. Try ecg_data/<relative_path>
    2. If missing, try input_segments/<relative_path>.json
    """
    # Default location (uploads)
    file_path = DATA_ROOT_DIR / relative_path

    # Fallback for dataset JSONs imported via import_json_segments_to_sql
    if not file_path.exists():
        alt1 = DATASET_JSON_DIR / (relative_path + ".json")
        alt2 = DATASET_JSON_DIR / relative_path  # in case filename already has .json
        if alt1.exists():
            file_path = alt1
        elif alt2.exists():
            file_path = alt2
        else:
            raise FileNotFoundError(
                f"ECG file not found at {file_path} or {alt1} or {alt2}"
            )

    full_signal, original_fs = _load_data_from_json(file_path)
    full_signal = _preprocess(full_signal, original_fs)

    start = segment_index * SEGMENT_LENGTH
    end = (segment_index + 1) * SEGMENT_LENGTH
    segment = full_signal[start:end]
    return segment.tolist()


# =========================================================
# Flask Routes
# =========================================================

@app.route("/")
def index():
    """
    Main dashboard view – loads first segment from SQL.
    """
    row = db_service.fetch_one("SELECT MIN(segment_id) FROM ecg_features_annotatable;")
    first_segment_id = row[0] if row and row[0] else 1

    load_segment_id = request.args.get("load_segment_id", first_segment_id)

    try:
        file_list = sorted([p.name for p in DATA_ROOT_DIR.iterdir() if p.is_file()])
    except Exception:
        file_list = []

    all_segments = db_service.get_all_segments()

    return render_template(
        "index.html",
        file_list=file_list,
        all_segments=all_segments,
        initial_segment_id=int(load_segment_id),
        TARGET_FS=TARGET_FS,
    )


@app.route("/upload_and_process", methods=["POST"])
def upload_and_process():
    """
    Optional: upload a JSON ECG and immediately index segments into the DB.
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(url_for("index"))

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = DATA_ROOT_DIR / filename

    try:
        file.save(filepath)
        filename_key = process_and_save_record(filepath)
        new_segment_id = db_service.get_first_segment_id_by_filename(filename_key)
        if not new_segment_id:
            new_segment_id = 1
        return redirect(f"/?load_segment_id={new_segment_id}")
    except Exception as e:
        return (
            f"ERROR: Processing or Database Insertion Failed: {e}. "
            "Please ensure the JSON file is valid.",
            500,
        )


# =========================================================
# XAI Clinical Explanation Endpoint (Option A)
# =========================================================

@app.route("/api/xai/<int:segment_id>")
def api_xai(segment_id: int):
    """
    Standardized Decision Engine & XAI Endpoint.
    Leverages pre-computed results from ecg_segments if available.
    """
    # 1. Try to fetch from the NEW table first (Optimized path)
    new_data = db_service.get_segment_new(segment_id)
    if new_data and new_data.get("events_json"):
        # We found pre-computed or manually annotated results!
        data = new_data["events_json"]
        
        if isinstance(data, dict):
            # Modern structured format
            events_list = data.get("events", [])
            bg_rhythm = new_data.get("background_rhythm") or "Sinus Rhythm"
            
            # RE-ARBITRATE: Handle Sinus Veto etc.
            from decision_engine.models import Event, EventCategory, DisplayState
            from decision_engine.rules import apply_display_rules
            
            event_objs = []
            for e_dict in events_list:
                # Type recovery and DEFAULTing for missing fields
                if "event_id" not in e_dict: e_dict["event_id"] = str(uuid.uuid4())
                if "start_time" not in e_dict: e_dict["start_time"] = 0.0
                if "end_time" not in e_dict: e_dict["end_time"] = 0.0
                if "event_type" not in e_dict: e_dict["event_type"] = "Unknown"
                
                # Dynamic Category Recovery
                if "event_category" not in e_dict:
                    etype = e_dict["event_type"]
                    if etype in ["PVC", "PAC", "Bigeminy", "Trigeminy", "Couplet"]:
                        e_dict["event_category"] = EventCategory.ECTOPY
                    else:
                        e_dict["event_category"] = EventCategory.RHYTHM
                elif isinstance(e_dict["event_category"], str):
                    e_dict["event_category"] = EventCategory(e_dict["event_category"])
                    
                if "display_state" in e_dict and isinstance(e_dict["display_state"], str):
                    e_dict["display_state"] = DisplayState(e_dict["display_state"])
                
                valid_keys = Event.__annotations__.keys()
                e_filtered = {k: v for k, v in e_dict.items() if k in valid_keys}
                event_objs.append(Event(**e_filtered))
            
            final_display = apply_display_rules(bg_rhythm, event_objs)
            data["final_display_events"] = [e.__dict__ for e in final_display]
            response = data
        elif isinstance(data, list):
            # Legacy list format
            bg_rhythm = new_data.get("background_rhythm") or "Sinus Rhythm"
            
            # Optional: Re-arbitrate legacy list too? Yes, for consistency.
            from decision_engine.models import Event, EventCategory, DisplayState
            from decision_engine.rules import apply_display_rules
            
            event_objs = []
            for e_dict in data:
                # Type recovery and DEFAULTing for missing fields
                if "event_id" not in e_dict: e_dict["event_id"] = str(uuid.uuid4())
                if "start_time" not in e_dict: e_dict["start_time"] = 0.0
                if "end_time" not in e_dict: e_dict["end_time"] = 0.0
                if "event_type" not in e_dict: e_dict["event_type"] = "Unknown"
                
                # Dynamic Category Recovery
                if "event_category" not in e_dict:
                    etype = e_dict["event_type"]
                    if etype in ["PVC", "PAC", "Bigeminy", "Trigeminy", "Couplet"]:
                        e_dict["event_category"] = EventCategory.ECTOPY
                    else:
                        e_dict["event_category"] = EventCategory.RHYTHM
                elif isinstance(e_dict["event_category"], str):
                    e_dict["event_category"] = EventCategory(e_dict["event_category"])
                    
                if "display_state" in e_dict and isinstance(e_dict["display_state"], str):
                    e_dict["display_state"] = DisplayState(e_dict["display_state"])
                
                valid_keys = Event.__annotations__.keys()
                e_filtered = {k: v for k, v in e_dict.items() if k in valid_keys}
                event_objs.append(Event(**e_filtered))
            
            final_display = apply_display_rules(bg_rhythm, event_objs)
            
            response = {
                "segment_index": new_data["segment_index"],
                "segment_state": new_data["segment_state"] or "ANALYZED",
                "background_rhythm": bg_rhythm,
                "events": data,
                "final_display_events": [e.__dict__ for e in final_display],
                "explanation": "Loaded from clinical workstation ground-truth records."
            }
        else:
            # Fallback for unexpected data types
            response = data
            
        return jsonify(response)

    # 2. Legacy Fallback (On-the-fly calculation)
    seg = db_service.get_segment_data(segment_id)
    if not seg:
        return jsonify({"error": "Segment not found"}), 404

    raw_signal = seg.get("raw_signal")
    if not raw_signal:
        raw_signal = _load_and_segment_raw_data(seg["filename"], seg["segment_index"])
    
    segment_np = np.array(raw_signal, dtype=np.float32)
    features = seg.get("features_json") or {}
    
    from signal_processing.artifact_detection import check_signal_quality
    quality = check_signal_quality(segment_np, TARGET_FS)
    ml_evidence = explain_segment(segment_np, features)
    
    ml_input = {
        "label": ml_evidence.get("rhythm", {}).get("label", "Unknown"),
        "confidence": ml_evidence.get("rhythm", {}).get("confidence", 0.0),
        "probs": ml_evidence.get("rhythm", {}).get("probs", []),
        "ectopy_label": ml_evidence.get("ectopy", {}).get("label", "None"),
        "ectopy_conf": ml_evidence.get("ectopy", {}).get("confidence", 0.0)
    }
    
    orchestrator = RhythmOrchestrator()
    decision = orchestrator.decide(
        ml_prediction=ml_input,
        clinical_features=features,
        sqi_result=quality,
        segment_index=seg["segment_index"]
    )
    
    decision.xai_notes.update(features) 
    explanation_text = explain_decision(decision)
    
    response = decision.to_dict()
    response["explanation"] = explanation_text
    response["saliency"] = ml_evidence.get("saliency", [])
    
    return jsonify(response)


# =========================================================
# Segment Fetch (ECG + Features + Annotation) for Dashboard
# =========================================================



@app.route("/api/segment/<int:segment_id>")
def get_segment_api(segment_id: int):
    """
    Fetch all necessary info for a specific segment ID.
    Prioritizes the optimized ecg_segments table.
    """
    # 1. Try NEW table
    meta = db_service.get_segment_new(segment_id)
    if not meta:
        # Fallback to legacy
        meta = db_service.get_segment_data(segment_id)
        if not meta:
            return jsonify({"error": "Segment not found"}), 404
        
        # Legacy load from signal files if not in JSONB
        raw_signal = meta.get("raw_signal")
        if not raw_signal:
            try:
                raw_signal = _load_and_segment_raw_data(meta["filename"], meta["segment_index"])
            except Exception as e:
                return jsonify({"error": f"Failed to load ECG: {e}"}), 500
    else:
        # Success from new table
        raw_signal = meta.get("raw_signal")

    features = meta.get("features_json") or {}
    mean_hr = float(features.get("mean_hr", 0.0))

    # Parse r-peaks from DB (ORIGINAL - for fallback)
    # But we want to OVERRIDE with NeuroKit for fresh display
    
    # 1. Calculate FRESH R-peaks on the fly
    try:
        r_peaks_arr = _detect_r_peaks_neurokit(np.array(raw_signal), TARGET_FS)
        # Convert to string for JSON
        r_peaks_for_frontend = ",".join(str(x) for x in r_peaks_arr)
    except Exception:
        r_peaks_arr = np.array([], dtype=int)
        r_peaks_for_frontend = ""

    # Recompute PR interval from the segment
    try:
        pr_interval_ms = _calculate_pr_interval(np.array(raw_signal), r_peaks_arr, TARGET_FS)
    except Exception:
        pr_interval_ms = 0.0

    # QRS width from features (robust to None/NaN)
    # QRS width from features (robust to None/NaN) -- RECALCULATED ON THE FLY with NeuroKit
    # Note: We prioritize recomputing it to fix old data issues in dashboard
    try:
        qrs_durations = _compute_qrs_durations(np.array(raw_signal), r_peaks_arr, TARGET_FS)
        if len(qrs_durations) > 0:
            qrs_mean_ms = float(np.mean(qrs_durations))
        else:
             # Fallback to stored features if NeuroKit returns nothing (rare)
             qrs_mean_ms = 0.0 
             qrs_list = features.get("qrs_durations_ms")
             if isinstance(qrs_list, list):
                q_clean = [float(v) for v in qrs_list if v is not None and not np.isnan(float(v))]
                if q_clean:
                    qrs_mean_ms = float(sum(q_clean)/len(q_clean))
    except Exception as e:
        qrs_mean_ms = float(features.get("mean_qrs", 0.0))

    return jsonify(
        {
            "segment_id": meta["segment_id"],
            "filename": meta["filename"],
            "segment_index": meta["segment_index"],
            "raw_signal": raw_signal,
            "fs": TARGET_FS,
            "length": SEGMENT_LENGTH,
            "arrhythmia_label": meta.get("arrhythmia_label"),
            "notes": meta.get("arrhythmia_text_notes", ""),
            "features": features,
            "mean_hr": mean_hr,
            "pr_interval": float(pr_interval_ms),
            "qrs_mean_ms": float(qrs_mean_ms),
            "r_peaks": r_peaks_for_frontend,
            "corrected_by": meta.get("corrected_by"),
            "corrected_at": meta.get("corrected_at"),
        }
    )


# =========================================================
# Annotation Save Endpoint
# =========================================================

# LEGACY SEGMENT ANNOTATION DROPPED IN FAVOR OF EVENT ANNOTATION

import uuid
@app.route("/api/annotate_beats", methods=["POST"])
def annotate_beats():
    """
    Bulk apply label to selected beat indices.
    Each beat is converted to a strict ±0.3s (0.6s total) window.
    """
    data = request.json
    segment_id = data.get("segment_id")
    beat_indices = data.get("beat_indices", [])
    label = data.get("label")

    if not segment_id or not beat_indices or not label:
        return jsonify({"error": "Missing required fields"}), 400

    # Strict Medical Window: ±0.3s around peak
    WINDOW_S = 0.3
    
    success_count = 0
    for idx_rel in beat_indices:
        # Convert relative index (samples) to relative time (seconds)
        peak_time = idx_rel / TARGET_FS
        
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": label,
            "start_time": max(0, peak_time - WINDOW_S),
            "end_time": min(10.0, peak_time + WINDOW_S),
            "annotation_source": "cardiologist",
            "annotation_status": "confirmed",
            "used_for_training": True
        }
        
        if db_service.save_event_to_db(segment_id, event):
            success_count += 1
    return jsonify({"status": "ok", "applied": success_count})


@app.route("/api/delete_annotation", methods=["POST"])
def delete_annotation():
    """
    Remove a specific annotation event.
    Payload: { "segment_id": int, "event_id": str }
    """
    data = request.json
    segment_id = data.get("segment_id")
    event_id = data.get("event_id")

    if not segment_id or not event_id:
        return jsonify({"error": "Missing segment_id or event_id"}), 400

    if db_service.delete_event(segment_id, event_id):
        return jsonify({"status": "ok", "message": "Annotation deleted"})
    else:
        return jsonify({"error": "Failed to delete or event not found"}), 500


# =========================================================
# Export Corrected Segments → retraining_data/ (JSON)
# =========================================================

# EXPORT TO JSON DROPPED - TRAINING USES DIRECT SQL CONNECTION


# =========================================================
# Next / Previous Navigation
# =========================================================

@app.route("/api/next_segment/<int:segment_id>")
def api_next_segment(segment_id: int):
    """
    Return the next available segment_id after the given one.
    If none, wrap to the minimum segment_id.
    """
    conn = db_service._connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MIN(segment_id)
                FROM ecg_features_annotatable
                WHERE segment_id > %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "next": int(row[0])})

            cur.execute("SELECT MIN(segment_id) FROM ecg_features_annotatable")
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "next": int(row[0])})

        return jsonify({"ok": False, "error": "No segments"}), 404
    finally:
        conn.close()


@app.route("/api/prev_segment/<int:segment_id>")
def api_prev_segment(segment_id: int):
    """
    Return the previous available segment_id before the given one.
    If none, wrap to the maximum segment_id.
    """
    conn = db_service._connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(segment_id)
                FROM ecg_features_annotatable
                WHERE segment_id < %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "prev": int(row[0])})

            cur.execute("SELECT MAX(segment_id) FROM ecg_features_annotatable")
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "prev": int(row[0])})

        return jsonify({"ok": False, "error": "No segments"}), 404
    finally:
        conn.close()


# =========================================================
# Retrain Model Endpoint (Button in UI)
# =========================================================

@app.route("/api/retrain_model", methods=["GET", "POST"])
def api_retrain_model():
    """
    Called by dashboard "Retrain Model Using Corrected Segments" button.

    Pipeline:
      1) export_corrected_segments()  -> retraining_data/
      2) run retrain_model.py         -> outputs/checkpoints/best_model.pth
      3) xai.reset_model()            -> reload new weights on next XAI call
    """
    try:
        # 🔒 ISSUE 4: Retraining Gate enforcement
        count = db_service.count_confirmed_cardiologist_events()
        if count < 10: 
            return jsonify({
                "error": f"Insufficient data. Need at least 10 cardiologist-confirmed events (Current: {count})."
            }), 400

        script_path = BASE_DIR / "models_training" / "retrain.py"
        
        with open("training_log.txt", "w") as log_file:
            subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR / "models_training")
            )

        return jsonify({"status": "ok", "message": "Training started in background! Check training_log.txt for progress."})
    except Exception as e:
        return jsonify({"error": str(e)})


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # Run on 0.0.0.0 so you can view from other machines in LAN if needed
    app.run(host="0.0.0.0", port=5000, debug=True)
