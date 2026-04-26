"""
=============================================================
WESAD Course Project — Step 1: Data Preprocessing
CSC 491/591 Ubiquitous Computing and Mobile Health
=============================================================
Pipeline:
  1. Load raw BVP (PPG) signal from Empatica E4
  2. Parse condition labels from SX_quest.csv
  3. Apply Butterworth bandpass filter (0.5–4 Hz, order 4)
  4. Segment into overlapping windows (30s, 50% overlap)
  5. Save processed segments + labels per subject
  6. Visualize raw vs filtered signal with condition bands

Usage:
  python step1_preprocessing.py
  (expects data folders at DATA_ROOT/SX/SX_E4_Data.zip
                           and DATA_ROOT/SX/SX_quest.csv)

  OR point BVP_PATH / QUEST_PATH directly at individual files.
=============================================================
"""

import os
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────
# CONFIGURATION — edit these paths as needed
# ─────────────────────────────────────────────
DATA_ROOT  = "./data"       # root folder containing S2/, S3/, … S17/
OUTPUT_DIR = "./processed"      # where .npy files and figures are saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subjects available in WESAD (S1 and S12 discarded due to sensor failure)
ALL_SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9",
                "S10","S11","S13","S14","S15","S16","S17"]

# Signal parameters
FS          = 64       # E4 BVP sampling rate (Hz)
LOWCUT      = 0.5      # Butterworth low cutoff (Hz) — removes baseline wander
HIGHCUT     = 4.0      # Butterworth high cutoff (Hz) — removes noise above 240 BPM
FILT_ORDER  = 4        # Filter order — balances sharpness vs phase distortion

# Segmentation parameters
WIN_SEC     = 30       # Window size in seconds
STEP_SEC    = 15       # Step size in seconds (50% overlap)

# WESAD label mapping (protocol conditions → integer label)
LABEL_MAP = {
    "Base":   1,   # Baseline
    "TSST":   2,   # Stress (Trier Social Stress Test)
    "Fun":    3,   # Amusement
    "Medi 1": 4,   # Meditation (both sessions → same label)
    "Medi 2": 4,
}
LABEL_NAMES = {1: "Baseline", 2: "Stress", 3: "Amusement", 4: "Meditation"}
COLORS      = {1: "#4C72B0",  2: "#DD4949", 3: "#55A868",  4: "#8172B2"}

# Subjects with known data quality issues (documented in SX_readme.txt)
# S5 Medi 1: subject may have fallen asleep — excluded from Meditation class
EXCLUDE_CONDITIONS = {
    "S5": ["Medi 1"],   # possible sleep artifact
}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def parse_time(t_str):
    """
    Convert WESAD time string 'MM.SS' to total seconds.
    e.g. '2.14'  → 2 min 14 sec → 134 s
         '26.24' → 26 min 24 sec → 1584 s
    NOTE: The decimal part is SECONDS (not fractional minutes).
    """
    t    = float(t_str)
    mins = int(t)
    secs = round((t - mins) * 100)
    return mins * 60 + secs


def load_bvp(bvp_path):
    """
    Load BVP signal from Empatica E4 CSV.
    Returns: (bvp_raw: np.ndarray, fs: float, start_unix: float)
    File format:
      Line 1: Unix timestamp of first sample
      Line 2: Sampling rate (Hz)
      Lines 3+: Signal values
    """
    with open(bvp_path) as f:
        lines = f.readlines()
    start_unix = float(lines[0].strip())
    fs         = float(lines[1].strip())
    bvp_raw    = np.array([float(l.strip()) for l in lines[2:]])
    return bvp_raw, fs, start_unix


def load_bvp_from_zip(zip_path):
    """Extract BVP.csv from SX_E4_Data.zip and load it."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open("BVP.csv") as f:
            lines = f.read().decode().splitlines()
    start_unix = float(lines[0].strip())
    fs         = float(lines[1].strip())
    bvp_raw    = np.array([float(l.strip()) for l in lines[2:]])
    return bvp_raw, fs, start_unix


def parse_labels(quest_path, subject_id, n_samples, fs):
    """
    Parse condition windows from SX_quest.csv.
    Returns:
      windows_info: list of (condition_name, start_sec, end_sec, label_id)
      labels_full:  np.ndarray of shape (n_samples,), 0=unlabeled
    """
    with open(quest_path) as f:
        raw_lines = [l.strip() for l in f.readlines()]

    order_line = next(l for l in raw_lines if "ORDER" in l)
    start_line = next(l for l in raw_lines if "START" in l)
    end_line   = next(l for l in raw_lines if "END"   in l)

    conditions = order_line.split(";")[1:]
    starts_raw = start_line.split(";")[1:]
    ends_raw   = end_line.split(";")[1:]

    excluded = EXCLUDE_CONDITIONS.get(subject_id, [])
    windows_info = []

    for cond, s, e in zip(conditions, starts_raw, ends_raw):
        cond = cond.strip()
        if cond not in LABEL_MAP or not s.strip():
            continue
        if cond in excluded:
            print(f"    [{subject_id}] EXCLUDING '{cond}' — data quality flag")
            continue
        s_sec = parse_time(s.strip())
        e_sec = parse_time(e.strip())
        windows_info.append((cond, s_sec, e_sec, LABEL_MAP[cond]))

    # Build sample-level label array
    labels_full = np.zeros(n_samples, dtype=int)
    for cond, s_sec, e_sec, label in windows_info:
        i_start = int(s_sec * fs)
        i_end   = min(int(e_sec * fs), n_samples)
        labels_full[i_start:i_end] = label

    return windows_info, labels_full


def bandpass_filter(signal, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=FILT_ORDER):
    """
    Apply zero-phase Butterworth bandpass filter.
    Uses filtfilt (forward + backward pass) → no phase distortion.
    """
    nyq  = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def segment_signal(bvp_filtered, labels_full, fs=FS,
                   win_sec=WIN_SEC, step_sec=STEP_SEC):
    """
    Slide overlapping windows across the filtered signal.
    Windows crossing condition boundaries or containing label=0
    (transient/unlabeled) are discarded.

    Returns:
      segments:   np.ndarray (N_windows, win_samples)
      seg_labels: np.ndarray (N_windows,)
      seg_times:  np.ndarray (N_windows,) — start time in seconds
    """
    n_samples    = len(bvp_filtered)
    win_samples  = int(win_sec  * fs)
    step_samples = int(step_sec * fs)

    segments, seg_labels, seg_times = [], [], []

    for start in range(0, n_samples - win_samples + 1, step_samples):
        end          = start + win_samples
        window_labels = labels_full[start:end]
        unique        = np.unique(window_labels)

        # Keep only pure single-condition windows
        if len(unique) != 1 or unique[0] == 0:
            continue

        segments.append(bvp_filtered[start:end])
        seg_labels.append(unique[0])
        seg_times.append(start / fs)

    return (np.array(segments),
            np.array(seg_labels),
            np.array(seg_times))


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_subject(subject_id, bvp_raw, bvp_filtered, time_axis,
                 windows_info, seg_labels, seg_times, fs=FS):
    """Generate and save 3 diagnostic figures per subject."""

    display_min = 15   # show first 15 minutes in overview

    # ── Fig 1: Overview (raw vs filtered, first 15 min) ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"WESAD {subject_id} — BVP Signal Overview",
                 fontsize=15, fontweight='bold')
    display_end = min(int(display_min * 60 * fs), len(bvp_raw))
    t_disp = time_axis[:display_end]

    for ax, sig, title in zip(
        axes,
        [bvp_raw[:display_end], bvp_filtered[:display_end]],
        ["Raw BVP Signal",
         f"Filtered BVP (Butterworth {LOWCUT}–{HIGHCUT} Hz, order {FILT_ORDER})"]
    ):
        ax.plot(t_disp / 60, sig, color="#555", linewidth=0.4, alpha=0.85)
        ax.set_ylabel("Amplitude (a.u.)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        for cond, s_sec, e_sec, label in windows_info:
            xs = max(s_sec, 0) / 60
            xe = min(e_sec, display_min * 60) / 60
            if xs < xe:
                ax.axvspan(xs, xe, alpha=0.18, color=COLORS[label])

    patches = [mpatches.Patch(color=COLORS[l], alpha=0.6,
                               label=LABEL_NAMES[l]) for l in sorted(LABEL_NAMES)]
    axes[0].legend(handles=patches, loc='upper right', fontsize=9)
    axes[1].set_xlabel("Time (minutes)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{subject_id}_fig1_overview.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 2: Raw vs filtered close-up (Baseline vs Stress, 5s) ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(f"WESAD {subject_id} — Raw vs Filtered (5s excerpt)",
                 fontsize=14, fontweight='bold')
    clip_len = int(5 * fs)

    for col, (lid, lname) in enumerate([(1, "Baseline"), (2, "Stress")]):
        idx_arr = np.where(seg_labels == lid)[0]
        if len(idx_arr) == 0:
            continue
        ss = int(seg_times[idx_arr[0]] * fs)
        tc = np.arange(clip_len) / fs

        axes[0, col].plot(tc, bvp_raw[ss:ss+clip_len], color="#999", linewidth=0.9)
        axes[0, col].set_title(f"{lname} — Raw",
                                fontweight='bold', color=COLORS[lid])
        axes[0, col].set_ylabel("Amplitude")

        axes[1, col].plot(tc, bvp_filtered[ss:ss+clip_len],
                          color=COLORS[lid], linewidth=1.1)
        axes[1, col].set_title(f"{lname} — Filtered",
                                fontweight='bold', color=COLORS[lid])
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{subject_id}_fig2_raw_vs_filtered.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 3: Window distribution ──
    fig, ax = plt.subplots(figsize=(7, 4))
    label_ids = sorted(LABEL_NAMES.keys())
    counts    = [np.sum(seg_labels == l) for l in label_ids]
    bars = ax.bar([LABEL_NAMES[l] for l in label_ids], counts,
                  color=[COLORS[l] for l in label_ids],
                  edgecolor='white', width=0.55)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                str(cnt), ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_title(f"Segment Distribution — {subject_id}",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Windows")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{subject_id}_fig3_distribution.png",
                dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────

def process_subject(subject_id):
    """Full preprocessing pipeline for one subject. Returns (segments, labels)."""
    print(f"\n  Processing {subject_id}...")

    subj_dir   = os.path.join(DATA_ROOT, subject_id)
    zip_path   = os.path.join(subj_dir, f"{subject_id}_E4_Data.zip")
    quest_path = os.path.join(subj_dir, f"{subject_id}_quest.csv")

    # Load BVP
    bvp_raw, fs, start_unix = load_bvp_from_zip(zip_path)
    n_samples  = len(bvp_raw)
    time_axis  = np.arange(n_samples) / fs
    print(f"    BVP: {n_samples:,} samples, {n_samples/fs/60:.1f} min @ {fs} Hz")

    # Parse labels
    windows_info, labels_full = parse_labels(quest_path, subject_id, n_samples, fs)
    for cond, s, e, lbl in windows_info:
        print(f"    {cond:<8} → label {lbl}  [{s}s–{e}s]  ({(e-s)/60:.1f} min)")

    labeled_pct = np.sum(labels_full > 0) / n_samples * 100
    print(f"    Labeled: {np.sum(labels_full>0):,} samples ({labeled_pct:.1f}%)")

    # Filter
    bvp_filtered = bandpass_filter(bvp_raw, LOWCUT, HIGHCUT, fs, FILT_ORDER)

    # Segment
    segments, seg_labels, seg_times = segment_signal(
        bvp_filtered, labels_full, fs, WIN_SEC, STEP_SEC)
    print(f"    Windows: {len(segments)} total")
    for lid, lname in LABEL_NAMES.items():
        print(f"      {lname}: {np.sum(seg_labels==lid)}")

    # Visualize
    plot_subject(subject_id, bvp_raw, bvp_filtered, time_axis,
                 windows_info, seg_labels, seg_times, fs)

    return segments, seg_labels


def main():
    print("=" * 60)
    print("  WESAD Preprocessing Pipeline")
    print(f"  Subjects: {ALL_SUBJECTS}")
    print(f"  Filter  : Butterworth {LOWCUT}–{HIGHCUT} Hz, order {FILT_ORDER}")
    print(f"  Window  : {WIN_SEC}s, step {STEP_SEC}s (50% overlap)")
    print("=" * 60)

    all_segments, all_labels, all_subjects = [], [], []

    for subj in ALL_SUBJECTS:
        subj_dir = os.path.join(DATA_ROOT, subj)
        if not os.path.isdir(subj_dir):
            print(f"  SKIP {subj} — folder not found")
            continue

        segs, lbls = process_subject(subj)
        all_segments.append(segs)
        all_labels.append(lbls)
        all_subjects.extend([subj] * len(lbls))

        # Save per-subject
        np.save(f"{OUTPUT_DIR}/{subj}_segments.npy", segs)
        np.save(f"{OUTPUT_DIR}/{subj}_labels.npy",   lbls)

    # Combine all subjects
    all_segments = np.vstack(all_segments)
    all_labels   = np.concatenate(all_labels)
    all_subjects = np.array(all_subjects)

    np.save(f"{OUTPUT_DIR}/all_segments.npy", all_segments)
    np.save(f"{OUTPUT_DIR}/all_labels.npy",   all_labels)
    np.save(f"{OUTPUT_DIR}/all_subjects.npy", all_subjects)

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total windows : {len(all_labels):,}")
    print(f"  Window shape  : {all_segments.shape}")
    for lid, lname in LABEL_NAMES.items():
        cnt = np.sum(all_labels == lid)
        print(f"  {lname:<14}: {cnt:,} ({100*cnt/len(all_labels):.1f}%)")
    print(f"\n  Saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
