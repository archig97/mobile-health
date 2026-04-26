"""
=============================================================
WESAD Course Project — Step 2c: 1D CNN (End-to-End)
CSC 491/591 Ubiquitous Computing and Mobile Health
=============================================================
Approach 1 (End-to-End):
  Raw PPG windows fed directly into a 1D CNN.
  The network learns its own feature representations
  from the raw waveform without hand-crafted features.

Architecture:
  Input      (1920 × 1)  — 30s @ 64 Hz
  Conv1D(32, k=7) + BN + ReLU + MaxPool(4)  → (480 × 32)
  Conv1D(64, k=5) + BN + ReLU + MaxPool(4)  → (120 × 64)
  Conv1D(128,k=3) + BN + ReLU + GlobalAvgPool → (128,)
  Dense(64) + Dropout(0.4)
  Dense(4)  + Softmax  → class probabilities

Design rationale:
  - Hierarchical Conv1D blocks extract features at different temporal scales
    (k=7 captures ~110ms, k=5 captures ~78ms, k=3 captures ~47ms)
  - BatchNorm stabilizes training across subjects with different physiology
  - GlobalAveragePooling reduces parameters vs Flatten (less overfitting)
  - Dropout(0.4) prevents overfitting given limited subject count
  - class_weight='balanced' handles Amusement underrepresentation
  - EarlyStopping(patience=5) prevents overfitting to training subjects

Evaluation: LOSO Cross-Validation (same protocol as classical models)

Requirements: tensorflow >= 2.x  OR  keras
=============================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers, models, callbacks

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PROCESSED_DIR = "./processed"
OUTPUT_DIR    = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES  = ["Baseline", "Stress", "Amusement", "Meditation"]
N_CLASSES    = 4
FS           = 64
WIN_SAMPLES  = 1920   # 30s × 64 Hz

# Training hyperparameters
EPOCHS       = 50
BATCH_SIZE   = 64
VAL_SPLIT    = 0.10
PATIENCE     = 10     # EarlyStopping patience

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

def build_cnn(input_len=WIN_SAMPLES, n_classes=N_CLASSES):
    """
    1D CNN for PPG emotion classification.

    Layer-by-layer breakdown:
      Input:              (batch, 1920, 1)
      Conv1D(32, k=7):    learns low-level wave features (~110ms receptive field)
      BatchNorm + ReLU:   normalize + nonlinearity
      MaxPool(4):         downsample (batch, 480, 32)
      Conv1D(64, k=5):    mid-level features (combined beat shape)
      BatchNorm + ReLU
      MaxPool(4):         (batch, 120, 64)
      Conv1D(128, k=3):   high-level features (HRV patterns)
      BatchNorm + ReLU
      GlobalAvgPool:      (batch, 128) — channel-wise averaging, position-invariant
      Dense(64) + ReLU:   learned combination of CNN features
      Dropout(0.4):       regularization
      Dense(4) + Softmax: class probabilities
    """
    inp = layers.Input(shape=(input_len, 1), name="ppg_input")

    # Block 1 — low-level features
    x = layers.Conv1D(32, kernel_size=7, padding='same', name='conv1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling1D(4, name='pool1')(x)

    # Block 2 — mid-level features
    x = layers.Conv1D(64, kernel_size=5, padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.MaxPooling1D(4, name='pool2')(x)

    # Block 3 — high-level features
    x = layers.Conv1D(128, kernel_size=3, padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)

    # Classifier head
    x   = layers.Dense(64, activation='relu', name='dense1')(x)
    x   = layers.Dropout(0.4, name='dropout')(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = models.Model(inp, out, name="1D_CNN_PPG")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def preprocess_cnn_input(segments):
    """
    Per-window z-score normalization.
    Removes absolute amplitude differences between subjects
    (different skin tones, sensor contact → different PPG amplitudes).
    """
    X = segments.copy().astype(np.float32)
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1,  keepdims=True) + 1e-8
    X   = (X - mu) / std
    return X[..., np.newaxis]   # (N, 1920, 1)


# ─────────────────────────────────────────────
# LOSO CROSS-VALIDATION
# ─────────────────────────────────────────────

def loso_cnn(X, y, subjects, verbose=True):
    """
    Leave-One-Subject-Out CV for the 1D CNN.
    A fresh model is created and trained for each fold.
    """
    unique_subjects = sorted(np.unique(subjects))
    all_true, all_pred = [], []
    per_subj_results   = {}

    if verbose:
        print(f"\n  1D CNN — LOSO ({len(unique_subjects)} folds)")
        print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, EarlyStop patience: {PATIENCE}")
        print(f"  {'Subject':<8} {'N_test':>7} {'Acc':>7} {'Macro F1':>10}")
        print(f"  {'-'*36}")

    for fold_i, test_subj in enumerate(unique_subjects):
        train_mask = subjects != test_subj
        test_mask  = subjects == test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # Class weights for this fold's training set
        cw_vals = compute_class_weight('balanced',
                                        classes=np.unique(y_train),
                                        y=y_train)
        cw_dict = dict(enumerate(cw_vals))

        # Fresh model each fold
        model = build_cnn(input_len=X.shape[1], n_classes=N_CLASSES)

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=0
        )

        model.fit(
            X_train, y_train,
            validation_split=VAL_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=cw_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
        all_true.extend(y_test)
        all_pred.extend(preds)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='macro')
        per_subj_results[test_subj] = {'acc': acc, 'f1': f1, 'n': int(test_mask.sum())}

        if verbose:
            print(f"  {test_subj:<8} {test_mask.sum():>7} {acc:>7.3f} {f1:>10.3f}")

        tf.keras.backend.clear_session()   # free GPU/CPU memory

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1  = f1_score(all_true, all_pred, average='macro')

    if verbose:
        print(f"  {'-'*36}")
        print(f"  {'OVERALL':<8} {len(all_true):>7} "
              f"{overall_acc:>7.3f} {overall_f1:>10.3f}")

    return all_true, all_pred, per_subj_results


# ─────────────────────────────────────────────
# ARCHITECTURE DIAGRAM
# ─────────────────────────────────────────────

def plot_architecture(filepath):
    """Render a visual diagram of the CNN architecture."""
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.set_facecolor('#F8F9FA'); fig.patch.set_facecolor('#F8F9FA')
    ax.axis('off')

    arch = [
        ("Input\n1920 × 1\n30s PPG\n@ 64 Hz",   "#6C757D"),
        ("Conv1D(32)\nkernel=7\nBN + ReLU\nMaxPool(4)\n→ 480×32", "#1C7293"),
        ("Conv1D(64)\nkernel=5\nBN + ReLU\nMaxPool(4)\n→ 120×64", "#1C7293"),
        ("Conv1D(128)\nkernel=3\nBN + ReLU\nGlobalAvg\n→ 128",    "#065A82"),
        ("Dense(64)\nReLU\nDropout\n(0.4)",       "#028090"),
        ("Dense(4)\nSoftmax\n→ 4 classes",         "#DD4949"),
    ]

    n = len(arch)
    w = 0.132; gap = 0.012

    for i, (label, color) in enumerate(arch):
        x0 = i * (w + gap) + 0.01
        rect = plt.Rectangle((x0, 0.10), w, 0.80,
                               transform=ax.transAxes, color=color,
                               clip_on=False, zorder=2, linewidth=0, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x0 + w/2, 0.50, label,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                zorder=3, multialignment='center', linespacing=1.4)
        if i < n - 1:
            x_start = x0 + w + 0.003
            x_end   = x0 + w + gap - 0.003
            ax.annotate('',
                        xy=(x_end, 0.50), xytext=(x_start, 0.50),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='-|>',
                                        color='#333333', lw=2.0,
                                        mutation_scale=15))

    ax.set_title("1D CNN Architecture — End-to-End PPG Emotion Recognition",
                 fontsize=13, fontweight='bold', pad=18, color='#212529')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WESAD 1D CNN — LOSO Cross-Validation")
    print("=" * 60)

    segments = np.load(f"{PROCESSED_DIR}/all_segments.npy")
    labels   = np.load(f"{PROCESSED_DIR}/all_labels.npy")
    subjects = np.load(f"{PROCESSED_DIR}/all_subjects.npy")

    # Map labels 1–4 → 0–3 for Keras
    label_map  = {1: 0, 2: 1, 3: 2, 4: 3}
    labels_enc = np.array([label_map[l] for l in labels])

    print(f"\n  Loaded   : {segments.shape[0]:,} windows, shape {segments.shape}")
    print(f"  Subjects : {sorted(np.unique(subjects))}")

    # Preprocess
    X = preprocess_cnn_input(segments)
    print(f"  CNN input: {X.shape}  (normalized, channel dim added)")

    # Print model summary once
    model_demo = build_cnn()
    model_demo.summary()
    tf.keras.backend.clear_session()

    # Architecture diagram
    plot_architecture(f"{OUTPUT_DIR}/cnn_architecture.png")

    # LOSO
    cnn_true, cnn_pred, cnn_subj = loso_cnn(X, labels_enc, subjects)

    # Report
    print(f"\n{'='*50}")
    print(f"  1D CNN — Classification Report")
    print(f"{'='*50}")
    print(classification_report(cnn_true, cnn_pred,
                                 target_names=CLASS_NAMES, digits=3))

    # Confusion matrix
    cm      = confusion_matrix(cnn_true, cnn_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Row Proportion', 'shrink': 0.85}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11, fontweight='bold')
    ax.set_ylabel("True",      fontsize=11, fontweight='bold')
    ax.set_title(
        f"1D CNN — Confusion Matrix (LOSO)\n"
        f"Acc={accuracy_score(cnn_true,cnn_pred):.3f}  "
        f"Macro F1={f1_score(cnn_true,cnn_pred,average='macro'):.3f}",
        fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=30); ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    cm_path = f"{OUTPUT_DIR}/cm_cnn.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {cm_path}")

    # Save
    np.save(f"{OUTPUT_DIR}/cnn_true.npy", cnn_true)
    np.save(f"{OUTPUT_DIR}/cnn_pred.npy", cnn_pred)

    cnn_results = {
        "accuracy":     float(accuracy_score(cnn_true, cnn_pred)),
        "macro_f1":     float(f1_score(cnn_true, cnn_pred, average='macro')),
        "per_class_f1": f1_score(cnn_true, cnn_pred, average=None).tolist(),
        "per_subject":  {k: {"acc": v["acc"], "f1": v["f1"]}
                         for k, v in cnn_subj.items()},
    }
    with open(f"{OUTPUT_DIR}/cnn_results.json", "w") as f:
        json.dump(cnn_results, f, indent=2)

    print(f"\n  Results saved to {OUTPUT_DIR}/")
    print(f"  CNN Accuracy : {cnn_results['accuracy']:.4f}")
    print(f"  CNN Macro F1 : {cnn_results['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
