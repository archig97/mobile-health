"""
=============================================================
WESAD Course Project — Step 2a: HRV Feature Extraction
CSC 491/591 Ubiquitous Computing and Mobile Health
=============================================================
Extracts 12 hand-crafted HRV and PPG features per window:

Time-domain HRV:
  mean_hr   — Mean heart rate (BPM)
  sdnn      — Std deviation of NN intervals (ms)
  rmssd     — Root mean square of successive differences (ms)
  pnn50     — % of successive differences > 50ms
  mean_ibi  — Mean inter-beat interval (ms)
  cv_ibi    — Coefficient of variation of IBI

PPG morphology:
  ppg_mean  — Mean amplitude
  ppg_std   — Std of amplitude
  ppg_range — Peak-to-peak range

Frequency-domain HRV (via IBI resampled at 4 Hz → FFT):
  lf_power  — Low-frequency power (0.04–0.15 Hz) — sympathetic + parasympathetic
  hf_power  — High-frequency power (0.15–0.4 Hz) — parasympathetic (HF = breathing)
  lf_hf     — LF/HF ratio — sympathovagal balance; elevated during stress

Usage:
  python step2a_feature_extraction.py
  (reads from OUTPUT_DIR set in step1_preprocessing.py)
=============================================================
"""

import os
import numpy as np
from scipy.signal import find_peaks

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PROCESSED_DIR = "./processed"   # output of step1_preprocessing.py
OUTPUT_DIR    = "./processed"
FS            = 64              # BVP sampling rate (Hz)
IBI_RESAMPLE  = 4               # Hz for IBI resampling before FFT


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def detect_peaks(window, fs=FS):
    """
    Detect systolic peaks in a PPG window using scipy find_peaks.
    min_distance = 0.4s (corresponds to max HR of 150 BPM).
    Returns array of peak indices.
    """
    peaks, _ = find_peaks(window,
                           distance=int(0.4 * fs),  # min 0.4s between peaks
                           height=0)                 # peaks must be positive
    return peaks


def compute_ibi(peaks, fs=FS):
    """Inter-Beat Intervals in milliseconds from peak indices."""
    return np.diff(peaks) / fs * 1000.0


def compute_frequency_features(ibi_ms, resample_hz=IBI_RESAMPLE):
    """
    Estimate LF and HF power from the IBI series using FFT.
    Steps:
      1. Convert IBI series to a time-stamped sequence
      2. Resample to uniform 4 Hz grid (required for FFT)
      3. Apply FFT and integrate power in LF (0.04–0.15 Hz)
         and HF (0.15–0.4 Hz) bands
    Returns: lf_power, hf_power, lf_hf_ratio
    """
    if len(ibi_ms) < 4:
        return 0.0, 0.0, 0.0

    t_ibi     = np.cumsum(ibi_ms) / 1000.0          # seconds
    t_uniform = np.arange(t_ibi[0], t_ibi[-1], 1.0 / resample_hz)

    if len(t_uniform) < 8:
        return 0.0, 0.0, 0.0

    # Resample IBI to uniform grid
    ibi_uniform = np.interp(t_uniform, t_ibi, ibi_ms)
    ibi_detrend = ibi_uniform - np.mean(ibi_uniform)  # remove DC

    # FFT
    fft_vals = np.abs(np.fft.rfft(ibi_detrend))
    freqs    = np.fft.rfftfreq(len(ibi_detrend), 1.0 / resample_hz)

    # Power in each band (sum of squared amplitudes)
    lf_mask  = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask  = (freqs >= 0.15) & (freqs < 0.40)

    lf_power = float(np.sum(fft_vals[lf_mask] ** 2))
    hf_power = float(np.sum(fft_vals[hf_mask] ** 2))
    lf_hf    = lf_power / (hf_power + 1e-8)

    return lf_power, hf_power, lf_hf


def extract_features(window, fs=FS):
    """
    Extract all 12 HRV + PPG features from a single window.
    Returns dict of features, or None if insufficient peaks detected.

    Feature descriptions:
      mean_hr   — 60000 / mean(IBI)  [BPM]
      sdnn      — std(IBI)           [ms]   overall HRV measure
      rmssd     — sqrt(mean(diff(IBI)^2))  [ms]  short-term HRV
      pnn50     — % of |diff(IBI)| > 50ms  [%]   parasympathetic indicator
      mean_ibi  — mean(IBI)          [ms]
      cv_ibi    — std/mean of IBI    [dimensionless]
      ppg_mean  — mean(window)
      ppg_std   — std(window)
      ppg_range — max(window) - min(window)
      lf_power  — LF spectral power  [ms^2]
      hf_power  — HF spectral power  [ms^2]
      lf_hf     — LF/HF ratio        [dimensionless] ↑ in stress
    """
    peaks = detect_peaks(window, fs)
    if len(peaks) < 3:
        return None   # too few beats to compute HRV reliably

    ibi = compute_ibi(peaks, fs)

    # Time-domain
    diff_ibi = np.diff(ibi)
    feats = {
        'mean_hr':   60000.0 / np.mean(ibi),
        'sdnn':      float(np.std(ibi)),
        'rmssd':     float(np.sqrt(np.mean(diff_ibi ** 2))),
        'pnn50':     float(np.mean(np.abs(diff_ibi) > 50) * 100),
        'mean_ibi':  float(np.mean(ibi)),
        'cv_ibi':    float(np.std(ibi) / (np.mean(ibi) + 1e-8)),
    }

    # PPG morphology
    feats['ppg_mean']  = float(np.mean(window))
    feats['ppg_std']   = float(np.std(window))
    feats['ppg_range'] = float(np.ptp(window))

    # Frequency-domain
    lf, hf, ratio = compute_frequency_features(ibi)
    feats['lf_power'] = lf
    feats['hf_power'] = hf
    feats['lf_hf']    = ratio

    return feats


def extract_all(segments, verbose=True):
    """
    Extract features from all windows in the dataset.
    Windows where < 3 peaks are detected are dropped.

    Returns:
      X          — np.ndarray (N_valid, n_features)
      valid_idx  — indices of valid windows (subset of 0..len(segments)-1)
      feat_names — list of feature names (length n_features)
    """
    if verbose:
        print(f"  Extracting features from {len(segments):,} windows...")

    feat_list, valid_idx = [], []

    for i, win in enumerate(segments):
        f = extract_features(win, FS)
        if f is not None:
            feat_list.append(list(f.values()))
            valid_idx.append(i)

    feat_names = list(extract_features(segments[0], FS).keys())
    X          = np.array(feat_list)
    valid_idx  = np.array(valid_idx)

    if verbose:
        print(f"  Valid windows : {len(valid_idx):,} / {len(segments):,}")
        print(f"  Feature names : {feat_names}")

    return X, valid_idx, feat_names


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WESAD Feature Extraction")
    print("=" * 60)

    segments = np.load(f"{PROCESSED_DIR}/all_segments.npy")
    labels   = np.load(f"{PROCESSED_DIR}/all_labels.npy")
    subjects = np.load(f"{PROCESSED_DIR}/all_subjects.npy")

    print(f"  Loaded: {segments.shape[0]:,} windows, shape {segments.shape}")

    X, valid_idx, feat_names = extract_all(segments, verbose=True)
    y    = labels[valid_idx]
    subj = subjects[valid_idx]

    # Save
    np.save(f"{OUTPUT_DIR}/X_features.npy",  X)
    np.save(f"{OUTPUT_DIR}/y_features.npy",  y)
    np.save(f"{OUTPUT_DIR}/subj_features.npy", subj)
    np.save(f"{OUTPUT_DIR}/valid_idx.npy",   valid_idx)

    import json
    with open(f"{OUTPUT_DIR}/feature_names.json", "w") as f:
        json.dump(feat_names, f)

    print(f"\n  Feature matrix : {X.shape}  (windows × features)")
    print(f"  Saved to       : {OUTPUT_DIR}/")

    # Basic stats per class
    LABEL_NAMES = {1:"Baseline",2:"Stress",3:"Amusement",4:"Meditation"}
    print("\n  Feature means by class:")
    header = f"  {'Feature':<12}" + "".join(f"  {LABEL_NAMES[l]:>12}" for l in sorted(LABEL_NAMES))
    print(header)
    for fi, fn in enumerate(feat_names):
        row = f"  {fn:<12}"
        for l in sorted(LABEL_NAMES):
            vals = X[y == (l-1), fi]  # y is 0-indexed after label_map
            row += f"  {np.mean(vals):>12.3f}"
        print(row)

    print("\n  Done.")


if __name__ == "__main__":
    main()
