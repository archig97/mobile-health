# WESAD Emotion Recognition — Code Guide
## CSC 491/591 Ubiquitous Computing and Mobile Health

---

## Project Overview

This codebase implements a full pipeline for 4-class emotion recognition
from PPG (photoplethysmography) signals using the WESAD dataset.

**Classes:** Baseline (1), Stress (2), Amusement (3), Meditation (4)
**Subjects:** 15 (S2–S17, excluding S1 and S12 — sensor failure)
**Signal:** Wrist PPG (BVP) from Empatica E4 @ 64 Hz
**Evaluation:** Leave-One-Subject-Out (LOSO) Cross-Validation

---

## File Structure

```
code/
├── step1_preprocessing.py      # Section 1: Data loading, filtering, segmentation
├── step2a_feature_extraction.py # Section 2: HRV feature extraction (12 features)
├── step2b_classical_models.py   # Section 2: SVM + Random Forest (LOSO)
├── step2c_cnn_model.py          # Section 2: 1D CNN end-to-end (LOSO)
└── step3_results_analysis.py    # Section 3: All result figures and analysis
```

---

## Dependencies

```bash
pip install numpy scipy matplotlib seaborn scikit-learn tensorflow
```

---

## How to Run

### Step 1: Preprocess all subjects
```bash
python step1_preprocessing.py
```
**Input:**  `./WESAD/SX/SX_E4_Data.zip` and `./WESAD/SX/SX_quest.csv`
**Output:** `./processed/all_segments.npy` (2942, 1920)
            `./processed/all_labels.npy`   (2942,)
            `./processed/all_subjects.npy` (2942,)
            Per-subject figures in `./processed/`

### Step 2a: Extract HRV features
```bash
python step2a_feature_extraction.py
```
**Output:** `./processed/X_features.npy` (2942, 12)
            `./processed/feature_names.json`

### Step 2b: Train classical models (SVM + RF)
```bash
python step2b_classical_models.py
```
**Output:** `./results/svm_true.npy`, `./results/svm_pred.npy`
            `./results/rf_true.npy`,  `./results/rf_pred.npy`
            `./results/classical_results.json`
            Confusion matrices + feature importance figures

### Step 2c: Train 1D CNN
```bash
python step2c_cnn_model.py
```
**Output:** `./results/cnn_true.npy`, `./results/cnn_pred.npy`
            `./results/cnn_results.json`
            CNN architecture diagram + confusion matrix

### Step 3: Generate analysis figures
```bash
python step3_results_analysis.py
```
**Output:** All comparison figures in `./results/`

---

## Key Design Decisions

### 1. Signal Filtering
- **Butterworth bandpass (0.5–4 Hz, order 4)**
- Low cutoff 0.5 Hz removes baseline wander from breathing/motion
- High cutoff 4.0 Hz removes noise (4 Hz = 240 BPM upper HR limit)
- `filtfilt` (zero-phase) — no temporal distortion of waveform

### 2. Segmentation
- **30-second windows** — long enough to compute HRV (≥5 heartbeats needed)
- **50% overlap (15s step)** — multiplies training samples without excessive redundancy
- Windows spanning condition boundaries are **discarded** (pure-label guarantee)
- Label 0 (transient/undefined) always excluded

### 3. Data Quality Handling
| Subject | Issue | Action |
|---------|-------|--------|
| S5      | Possible sleep during Medi 1 | Medi 1 excluded |
| S6      | Low TSST stress (interview-habituated) | Retained, documented |
| S9      | Felt ill on study day | Retained, documented |
| S15     | Disbelieved TSST cover story | Retained, documented |
| S8, S16 | Cold room during stress | Retained, documented |
| S2, S17 | RespiBAN TEMP issue | Irrelevant (BVP pipeline) |

### 4. LOSO Cross-Validation
- Split is **by subject**, not by window
- Prevents data leakage (windows from same subject are correlated)
- StandardScaler fit ONLY on training fold each time (no test leakage)
- Reflects real-world: model deployed on an unseen person

### 5. Class Imbalance
- Amusement has ~3× fewer windows than Baseline
- `class_weight='balanced'` used in all classifiers
- Macro F1 reported (treats all classes equally regardless of support)

---

## HRV Features Extracted (12 total)

| Feature    | Domain    | Description |
|------------|-----------|-------------|
| mean_hr    | Time      | Mean heart rate (BPM) |
| sdnn       | Time      | Std of NN intervals — overall HRV |
| rmssd      | Time      | Root mean square successive differences |
| pnn50      | Time      | % successive differences > 50ms |
| mean_ibi   | Time      | Mean inter-beat interval (ms) |
| cv_ibi     | Time      | Coefficient of variation of IBI |
| ppg_mean   | Amplitude | Mean PPG amplitude |
| ppg_std    | Amplitude | Std of PPG amplitude |
| ppg_range  | Amplitude | Peak-to-peak range |
| lf_power   | Frequency | 0.04–0.15 Hz power (sympathetic) |
| hf_power   | Frequency | 0.15–0.40 Hz power (parasympathetic) |
| lf_hf      | Frequency | LF/HF ratio (sympathovagal balance) |

---

## Results Summary (LOSO, 15 subjects)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| SVM (Feature-Based) | 35.9% | 0.323 |
| Random Forest (Feature-Based) | 40.0% | 0.304 |
| 1D CNN (End-to-End) | 36.3% | 0.184 |
| Chance baseline | 25.0% | — |

---

## References

1. Schmidt et al. (2018). Introducing WESAD, a Multimodal Dataset for
   Wearable Stress and Affect Detection. ICMI '18.
   https://doi.org/10.1145/3242969.3242985

2. Birjandtalab et al. (2016). A non-EEG biosignals dataset for assessment
   and visualization of neurological status.

3. Siirtola & Röning (2020). Feature Augmented Hybrid CNN for Stress
   Recognition Using Wrist-based PPG Sensor.
