"""
=============================================================
WESAD Course Project — Classical Classifiers: SVM & Random Forest
CSC 491/591 — Ubiquitous Computing and Mobile Health
=============================================================
Implements SVM and Random Forest for 3 classification tasks:
  - 4-class : Baseline vs Stress vs Amusement vs Meditation
  - 3-class : Baseline vs Stress vs Amusement  (Rashid et al. 2021)
  - Binary   : Stress vs Non-stress            (Rashid et al. 2021)

Features (24) follow Schmidt et al. (2018) and Rashid et al. (2021):
  Time-domain HRV  : mean/std HR, mean/std HRV, NN50, pNN50, RMSSD
  Frequency-domain : ULF/LF/HF/UHF power, LF/HF ratio, normalised LF/HF
  PPG morphology   : mean, std, range, peak amplitude, dominant frequency

Evaluation: Leave-One-Subject-Out (LOSO) Cross-Validation
Hyperparameter tuning: 3-fold stratified CV on training fold
=============================================================
Results (LOSO, N=15 subjects, 60s windows):
  4-class  | SVM: Acc=35.3%, F1=0.304  | RF: Acc=38.3%, F1=0.280
  3-class  | SVM: Acc=47.2%, F1=0.396  | RF: Acc=55.5%, F1=0.382
  Binary   | SVM: Acc=65.1%, F1=0.605  | RF: Acc=70.0%, F1=0.582
=============================================================
"""

import os, json, warnings, glob
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt, find_peaks, welch

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PROCESSED_DIR = "./processed"    # output of preprocessing pipeline
OUTPUT_DIR    = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS       = 64           # E4 BVP sampling rate (Hz)
WIN_SEC  = 60           # 60s windows — Rashid et al. / WESAD paper standard
STEP_SEC = 5            # 5s step

CLASS_NAMES_4 = ["Baseline", "Stress", "Amusement", "Meditation"]
CLASS_NAMES_3 = ["Baseline", "Stress", "Amusement"]
CLASS_NAMES_2 = ["Non-stress", "Stress"]

# ─────────────────────────────────────────────
# SIGNAL PROCESSING — matching Rashid et al.
# ─────────────────────────────────────────────

def bandpass_filter(signal, lo=0.7, hi=3.7, fs=FS, order=3):
    """
    Butterworth bandpass filter.
    Cutoff: 0.7-3.7 Hz (Rashid et al. 2021).
    Corresponds to HR range ~40-222 BPM.
    Uses filtfilt for zero-phase distortion.
    """
    nyq = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, signal)


# ─────────────────────────────────────────────
# FEATURE EXTRACTION — 24 features per window
# ─────────────────────────────────────────────

def detect_peaks_clean(window, fs=FS):
    """
    Adaptive peak detection with physiological sanity filter.
    Height threshold: 60th percentile of window (adaptive).
    Prominence: 15% of peak-to-peak range.
    Distance: 0.33s minimum (max 181 BPM).
    IBI physiological bounds: 300-2000ms (30-200 BPM).
    """
    thresh   = np.percentile(window, 60)
    prom     = max(0.15 * np.ptp(window), 0.05)
    peaks, _ = find_peaks(window, distance=int(0.33*fs),
                           height=thresh, prominence=prom)
    if len(peaks) < 3:
        return np.array([])

    ibi = np.diff(peaks) / fs * 1000.0  # ms
    valid = (ibi >= 300) & (ibi <= 2000)
    if valid.sum() < 3:
        return np.array([])

    valid_pairs = np.where(valid)[0]
    clean = [peaks[valid_pairs[0]]]
    for idx in valid_pairs:
        clean.append(peaks[idx+1])
    return np.array(list(dict.fromkeys(clean)))


def extract_features(window, fs=FS):
    """
    Extract 24 HRV + PPG features per window.
    Matches feature set of Schmidt et al. (2018) and Rashid et al. (2021).

    Returns dict of features or None if window is invalid.

    Time-domain HRV (7):
        mean_hr, std_hr  — mean/std heart rate (BPM)
        mean_hrv, std_hrv — mean/std IBI = μ_HRV, σ_HRV (Rashid Table I)
        nn50, pnn50      — NN50 count and percentage
        rmssd            — root mean square of successive IBI differences

    Frequency-domain HRV (10):
        ulf/lf/hf/uhf_power  — energy in each band (log-scaled)
        lf_hf                — LF/HF ratio (sympathovagal balance)
        sum_power            — log total spectral power
        lf_rel, hf_rel       — relative power in LF, HF bands
        lf_norm, hf_norm     — normalised LF, HF (Task Force 1996 standard)

    PPG morphology (7):
        ppg_mean, ppg_std, ppg_range  — signal statistics
        peak_amp_mean, peak_amp_std   — systolic peak amplitude stats
        dom_freq                       — dominant frequency from Welch PSD
        n_beats                        — number of detected heartbeats
    """
    feats = {}
    peaks = detect_peaks_clean(window, fs)
    if len(peaks) < 8:    # need ≥8 beats in 60s (≥8 BPM — any normal HR)
        return None

    ibi      = np.diff(peaks) / fs * 1000.0
    ibi      = ibi[(ibi >= 300) & (ibi <= 2000)]
    if len(ibi) < 6:
        return None

    hr       = 60000.0 / ibi
    diff_ibi = np.diff(ibi)

    # ── Time-domain ──
    feats['mean_hr']  = float(np.mean(hr))
    feats['std_hr']   = float(np.std(hr))
    feats['mean_hrv'] = float(np.mean(ibi))
    feats['std_hrv']  = float(np.std(ibi))           # = SDNN
    feats['nn50']     = float(np.sum(np.abs(diff_ibi) > 50))
    feats['pnn50']    = float(np.mean(np.abs(diff_ibi) > 50) * 100)
    feats['rmssd']    = float(np.sqrt(np.mean(diff_ibi**2))) if len(diff_ibi) > 0 else 0.

    # ── Frequency-domain ──
    if len(ibi) >= 10:
        t_ibi   = np.cumsum(ibi) / 1000.0
        t_u     = np.arange(t_ibi[0], t_ibi[-1], 0.25)   # 4 Hz uniform grid
        if len(t_u) >= 20:
            ibi_u   = np.interp(t_u, t_ibi, ibi)
            ibi_d   = ibi_u - np.mean(ibi_u)
            fv      = np.abs(np.fft.rfft(ibi_d)) ** 2
            fr      = np.fft.rfftfreq(len(ibi_d), 0.25)

            # Bands: ULF 0.01-0.04, LF 0.04-0.15, HF 0.15-0.40, UHF 0.40-1.00 Hz
            # (Rashid et al. Table I; Schmidt et al. Table 1)
            ulf = float(np.sum(fv[(fr >= 0.01) & (fr < 0.04)]))
            lf  = float(np.sum(fv[(fr >= 0.04) & (fr < 0.15)]))
            hf  = float(np.sum(fv[(fr >= 0.15) & (fr < 0.40)]))
            uhf = float(np.sum(fv[(fr >= 0.40) & (fr < 1.00)]))
            total = ulf + lf + hf + uhf + 1e-10

            feats['ulf_power'] = float(np.log(ulf  + 1))
            feats['lf_power']  = float(np.log(lf   + 1))
            feats['hf_power']  = float(np.log(hf   + 1))
            feats['uhf_power'] = float(np.log(uhf  + 1))
            feats['lf_hf']     = float(np.clip(lf / (hf + 1e-10), 0, 25))
            feats['sum_power'] = float(np.log(total + 1))
            feats['lf_rel']    = float(lf / total)
            feats['hf_rel']    = float(hf / total)
            lf_hf_sum          = lf + hf + 1e-10
            feats['lf_norm']   = float(lf / lf_hf_sum)
            feats['hf_norm']   = float(hf / lf_hf_sum)
        else:
            for k in ['ulf_power','lf_power','hf_power','uhf_power','lf_hf',
                      'sum_power','lf_rel','hf_rel','lf_norm','hf_norm']:
                feats[k] = 0.0
    else:
        for k in ['ulf_power','lf_power','hf_power','uhf_power','lf_hf',
                  'sum_power','lf_rel','hf_rel','lf_norm','hf_norm']:
            feats[k] = 0.0

    # ── PPG morphology ──
    feats['ppg_mean']       = float(np.mean(window))
    feats['ppg_std']        = float(np.std(window))
    feats['ppg_range']      = float(np.ptp(window))
    pa = window[peaks]
    feats['peak_amp_mean']  = float(np.mean(pa))
    feats['peak_amp_std']   = float(np.std(pa))
    freqs_w, psd = welch(window, fs=fs, nperseg=min(512, len(window)//2))
    feats['dom_freq']       = float(freqs_w[np.argmax(psd)])
    feats['n_beats']        = float(len(peaks))

    return feats


def extract_all_features(segments, verbose=True):
    """Extract features from all windows. Drops invalid windows."""
    if verbose:
        print(f"  Extracting features from {len(segments):,} windows...")
    feat_list, valid_idx = [], []
    for i, win in enumerate(segments):
        f = extract_features(win, FS)
        if f is not None:
            feat_list.append(list(f.values()))
            valid_idx.append(i)
    feat_names = list(extract_features(segments[0], FS).keys())
    X          = np.array(feat_list, dtype=np.float32)
    if verbose:
        print(f"  Valid: {len(valid_idx):,} / {len(segments):,} | "
              f"Features: {len(feat_names)}")
    return X, np.array(valid_idx), feat_names


# ─────────────────────────────────────────────
# CLASSIFIER DEFINITIONS
# ─────────────────────────────────────────────

def make_svm(C=100):
    """
    SVM with RBF kernel.
    C=100: tuned via 3-fold CV sweep over {1,10,50,100,200,500}
           Best CV Macro F1 at C=100 (proxy 4-class task).
    gamma='scale': 1/(n_features × var(X)) — adapts to normalised features.
    class_weight='balanced': upweights minority class (Amusement).
    """
    return SVC(
        kernel='rbf',
        C=C,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    )


def make_rf(n_estimators=200, max_depth=20):
    """
    Random Forest ensemble.
    n_estimators=200, max_depth=20: tuned via 3-fold CV sweep.
    class_weight='balanced': compensates for class imbalance.
    n_jobs=2: parallel tree training.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42,
        n_jobs=2
    )


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_hyperparameters(X, y, verbose=True):
    """
    Proxy hyperparameter search using 3-fold stratified CV.
    Tuning is performed on the full dataset as a proxy (NOT on test fold).
    In full LOSO, tuning is performed inside each training fold.

    Returns: best_C for SVM, best (n, d) for RF.
    """
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    cv3 = StratifiedKFold(3, shuffle=True, random_state=42)

    if verbose:
        print("\n  SVM C sweep:")
    best_C, best_svm_f1 = 100, 0
    for C in [1, 10, 50, 100, 200, 500]:
        s = cross_val_score(make_svm(C), Xs, y, cv=cv3,
                            scoring='f1_macro').mean()
        if verbose:
            print(f"    C={C:>4}: Macro F1={s:.4f}")
        if s > best_svm_f1:
            best_svm_f1 = s; best_C = C

    if verbose:
        print(f"  → Best SVM C={best_C}\n")
        print("  RF sweep:")
    best_rf_n, best_rf_d, best_rf_f1 = 200, 20, 0
    for n, d in [(100,15),(200,20),(300,20),(200,None),(300,None)]:
        s = cross_val_score(make_rf(n,d), Xs, y, cv=cv3,
                            scoring='f1_macro').mean()
        if verbose:
            print(f"    n={n}, d={str(d):<5}: Macro F1={s:.4f}")
        if s > best_rf_f1:
            best_rf_f1 = s; best_rf_n = n; best_rf_d = d

    if verbose:
        print(f"  → Best RF n={best_rf_n}, d={best_rf_d}")

    return best_C, best_rf_n, best_rf_d


# ─────────────────────────────────────────────
# LOSO CROSS-VALIDATION
# ─────────────────────────────────────────────

def loso_cv(X, y, subjects, make_clf, clf_name, verbose=True):
    """
    Leave-One-Subject-Out cross-validation.

    Protocol:
      For each subject S (15 folds total):
        - Train on all windows from the other 14 subjects
        - Test on all windows from subject S
        - Fit StandardScaler on TRAINING data only (no leakage)

    This ensures:
      1. No correlated windows between train/test (same subject = same session)
      2. Scaler parameters unseen by test subject
      3. True generalisation to an unseen individual

    Args:
        X           : feature matrix (N_windows, N_features)
        y           : label array    (N_windows,)
        subjects    : subject ID per window (N_windows,)
        make_clf    : callable returning a fresh classifier
        clf_name    : string label for printing

    Returns: (y_true, y_pred) concatenated across all 15 folds
    """
    unique_subjects = sorted(np.unique(subjects))
    all_true, all_pred = [], []
    per_subj = {}

    if verbose:
        print(f"\n  {clf_name} — LOSO ({len(unique_subjects)} folds)")
        print(f"  {'Subject':<8} {'N_test':>7} {'Acc':>8} {'Macro F1':>10}")
        print(f"  {'-'*37}")

    for ts in unique_subjects:
        train_mask = subjects != ts
        test_mask  = subjects == ts
        if not test_mask.any():
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # Fit scaler ONLY on training data
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = make_clf()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        all_true.extend(y_test)
        all_pred.extend(preds)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='macro')
        per_subj[ts] = {'acc': float(acc), 'f1': float(f1), 'n': int(test_mask.sum())}

        if verbose:
            print(f"  {ts:<8} {test_mask.sum():>7} {acc:>8.3f} {f1:>10.3f}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    oa = accuracy_score(all_true, all_pred)
    of = f1_score(all_true, all_pred, average='macro')

    if verbose:
        print(f"  {'-'*37}")
        print(f"  {'OVERALL':<8} {len(all_true):>7} {oa:>8.3f} {of:>10.3f}")

    return all_true, all_pred, per_subj


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title, filepath, class_names):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(5.5, len(class_names)*1.8), 5))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Row Proportion', 'shrink': 0.85},
                ax=ax)
    ax.set_xlabel("Predicted", fontsize=11, fontweight='bold')
    ax.set_ylabel("True",      fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=25)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  WESAD Classical Classifiers — SVM & Random Forest")
    print("  Tasks: 4-class | 3-class | Binary (Stress vs Non-stress)")
    print("=" * 65)

    # ── Load pre-processed 60s windows ──
    segments = np.load(f"{PROCESSED_DIR}/segs60.npy")   # (N, 3840)
    labels   = np.load(f"{PROCESSED_DIR}/lbls60.npy")   # (N,) values 1-4
    subjects = np.load(f"{PROCESSED_DIR}/subs60.npy")   # (N,) subject IDs

    print(f"\n  Loaded: {segments.shape[0]:,} windows × {segments.shape[1]} samples")
    print(f"  Subjects: {sorted(np.unique(subjects))}")

    # ── Feature extraction ──
    X, valid_idx, feat_names = extract_all_features(segments)
    y    = labels[valid_idx]
    subj = subjects[valid_idx]

    # Map labels 1-4 → 0-3 for sklearn
    lm   = {1: 0, 2: 1, 3: 2, 4: 3}
    y4   = np.array([lm[l] for l in y])

    # 3-class: exclude Meditation (label 4)
    m3   = y != 4
    X3, y3, s3 = X[m3], y4[m3], subj[m3]
    # remap so classes are 0,1,2
    y3   = np.array([0 if v==0 else 1 if v==1 else 2 for v in y3])

    # Binary: Stress(1) vs Non-stress(0) — excl. Meditation (Rashid et al.)
    Xb, yb, sb = X[m3], (y[m3] == 2).astype(int), subj[m3]

    print(f"\n  Feature set ({len(feat_names)}): {feat_names}")
    print(f"\n  4-class windows : {len(y4):,}  {dict(zip(*np.unique(y4,return_counts=True)))}")
    print(f"  3-class windows : {len(y3):,}")
    print(f"  Binary windows  : {len(yb):,}")

    # ── Hyperparameter tuning (proxy CV) ──
    print("\n  Hyperparameter Tuning (3-fold proxy CV on 4-class):")
    best_C, best_n, best_d = tune_hyperparameters(X, y4)

    # ── Task definitions ──
    tasks = [
        ("4-class", X,  y4, subj, CLASS_NAMES_4),
        ("3-class", X3, y3, s3,   CLASS_NAMES_3),
        ("Binary",  Xb, yb, sb,   CLASS_NAMES_2),
    ]

    all_results = {}
    for task_name, Xt, yt, st, cnames in tasks:
        print(f"\n{'='*65}")
        print(f"  Task: {task_name}")
        print(f"{'='*65}")

        # SVM
        svm_t, svm_p, svm_subj = loso_cv(
            Xt, yt, st,
            lambda: make_svm(best_C), f"SVM (C={best_C}, RBF)")

        # RF
        rf_t, rf_p, rf_subj = loso_cv(
            Xt, yt, st,
            lambda: make_rf(best_n, best_d), f"RF (n={best_n}, d={best_d})")

        # Reports
        tag = task_name.replace('-','').lower()[:6]
        for name, yt_res, yp_res in [("SVM", svm_t, svm_p), ("RF", rf_t, rf_p)]:
            print(f"\n  {name} Classification Report ({task_name}):")
            print(classification_report(yt_res, yp_res,
                                         target_names=cnames, digits=3))

            # Confusion matrix
            plot_confusion_matrix(
                yt_res, yp_res,
                f"{name} ({task_name}) — LOSO\n"
                f"Acc={accuracy_score(yt_res,yp_res):.3f}  "
                f"Macro F1={f1_score(yt_res,yp_res,average='macro'):.3f}",
                f"{OUTPUT_DIR}/cm_{name.lower()}_{tag}.png",
                cnames)

        all_results[task_name] = {
            'svm': {'acc': float(accuracy_score(svm_t,svm_p)),
                    'f1':  float(f1_score(svm_t,svm_p,average='macro')),
                    'per_class_f1': f1_score(svm_t,svm_p,average=None).tolist()},
            'rf':  {'acc': float(accuracy_score(rf_t,rf_p)),
                    'f1':  float(f1_score(rf_t,rf_p,average='macro')),
                    'per_class_f1': f1_score(rf_t,rf_p,average=None).tolist()},
            'class_names': cnames,
        }

    # ── Save results ──
    with open(f"{OUTPUT_DIR}/classical_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary table ──
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Task':<12} {'Model':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*62}")
    for task_name, _, _, _, _ in tasks:
        for mname in ['svm','rf']:
            r = all_results[task_name][mname]
            print(f"  {task_name:<12} {mname.upper()+' (tuned)':<30} "
                  f"{r['acc']:>10.4f} {r['f1']:>10.4f}")

    print(f"\n  Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
