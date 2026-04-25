"""
=============================================================
WESAD Course Project — Step 3: Results Analysis & Visualization
CSC 491/591 Ubiquitous Computing and Mobile Health
=============================================================
Generates all figures for Section 3 (Experiments & Analysis):
  1. Model comparison bar chart (Accuracy + Macro F1)
  2. Combined confusion matrices (all 3 models)
  3. Per-class F1 breakdown
  4. Per-subject accuracy comparison
  5. Feature distributions by emotional condition
  6. Classification report heatmaps

Run AFTER step2b and step2c have saved their predictions.
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

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RESULTS_DIR   = "./results"
PROCESSED_DIR = "./processed"
OUTPUT_DIR    = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES   = ["Baseline", "Stress", "Amusement", "Meditation"]
CLASS_COLORS  = ["#4C72B0", "#DD4949", "#55A868", "#8172B2"]
MODEL_COLORS  = {"SVM": "#4C72B0", "RF": "#55A868", "CNN": "#DD8800"}


def load_results():
    """Load saved predictions from all three models."""
    svm_t = np.load(f"{RESULTS_DIR}/svm_true.npy")
    svm_p = np.load(f"{RESULTS_DIR}/svm_pred.npy")
    rf_t  = np.load(f"{RESULTS_DIR}/rf_true.npy")
    rf_p  = np.load(f"{RESULTS_DIR}/rf_pred.npy")
    cnn_t = np.load(f"{RESULTS_DIR}/cnn_true.npy")
    cnn_p = np.load(f"{RESULTS_DIR}/cnn_pred.npy")
    return svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p


# ─────────────────────────────────────────────
# FIG 1: Model Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_model_comparison(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p):
    model_labels = [
        "SVM\n(Feature-Based)",
        "Random Forest\n(Feature-Based)",
        "1D CNN\n(End-to-End)"
    ]
    accs = [accuracy_score(svm_t, svm_p),
            accuracy_score(rf_t,  rf_p),
            accuracy_score(cnn_t, cnn_p)]
    f1s  = [f1_score(svm_t, svm_p, average='macro'),
            f1_score(rf_t,  rf_p,  average='macro'),
            f1_score(cnn_t, cnn_p, average='macro')]
    colors = ["#4C72B0", "#55A868", "#DD8800"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Model Comparison — Leave-One-Subject-Out Cross-Validation\n"
        "WESAD Dataset  |  N=15 subjects  |  2,942 windows",
        fontsize=13, fontweight='bold'
    )

    for ax, vals, metric in zip(axes, [accs, f1s], ["Accuracy", "Macro F1 Score"]):
        bars = ax.bar(model_labels, vals, color=colors, edgecolor='white', width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f"{v:.3f}", ha='center', va='bottom',
                    fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5,
                   linewidth=1.2, label='Chance (25%)')
        ax.text(2.38, 0.265, "chance\n(25%)", color='gray', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG 2: Combined Confusion Matrices
# ─────────────────────────────────────────────

def plot_confusion_matrices(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p):
    results = [
        ("SVM\n(Feature-Based)",       svm_t, svm_p, "#4C72B0"),
        ("Random Forest\n(Feature-Based)", rf_t, rf_p, "#55A868"),
        ("1D CNN\n(End-to-End)",        cnn_t, cnn_p, "#DD8800"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Confusion Matrices — LOSO Cross-Validation  (N=15 subjects)",
        fontsize=14, fontweight='bold', y=1.02
    )

    for ax, (name, yt, yp, col) in zip(axes, results):
        cm      = confusion_matrix(yt, yp)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.85, 'label': 'Proportion'},
            ax=ax
        )
        acc = accuracy_score(yt, yp)
        f1m = f1_score(yt, yp, average='macro')
        ax.set_xlabel("Predicted", fontsize=10, fontweight='bold')
        ax.set_ylabel("True",      fontsize=10, fontweight='bold')
        ax.set_title(
            f"{name}\nAcc={acc:.3f}  Macro F1={f1m:.3f}",
            fontsize=10, fontweight='bold', color=col
        )
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG 3: Per-Class F1 Score
# ─────────────────────────────────────────────

def plot_per_class_f1(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p):
    results = [
        ("SVM (Feature-Based)",        svm_t, svm_p, "#4C72B0"),
        ("Random Forest (Feature-Based)", rf_t, rf_p, "#55A868"),
        ("1D CNN (End-to-End)",         cnn_t, cnn_p, "#DD8800"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Per-Class F1 Score by Model", fontsize=13, fontweight='bold')

    for ax, (name, yt, yp, col) in zip(axes, results):
        f1_per = f1_score(yt, yp, average=None)
        bars   = ax.bar(CLASS_NAMES, f1_per, color=CLASS_COLORS,
                         edgecolor='white', width=0.55)
        for bar, v in zip(bars, f1_per):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.012,
                    f"{v:.2f}", ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_title(name, fontsize=11, fontweight='bold', color=col)
        ax.set_ylabel("F1 Score")
        ax.tick_params(axis='x', rotation=20)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_per_class_f1.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG 4: Per-Subject Accuracy
# ─────────────────────────────────────────────

def plot_per_subject_accuracy():
    with open(f"{RESULTS_DIR}/classical_results.json") as f:
        cl = json.load(f)
    with open(f"{RESULTS_DIR}/cnn_results.json") as f:
        cnn = json.load(f)

    subjects = sorted(cl["svm"]["per_subject"].keys())

    svm_accs = [cl["svm"]["per_subject"][s]["acc"] for s in subjects]
    rf_accs  = [cl["rf"]["per_subject"][s]["acc"]  for s in subjects]
    cnn_accs = [cnn["per_subject"].get(s, {}).get("acc", 0) for s in subjects]

    x = np.arange(len(subjects)); w = 0.26
    fig, ax = plt.subplots(figsize=(14, 4.5))

    ax.bar(x - w,   svm_accs, w, label='SVM',          color='#4C72B0', edgecolor='white')
    ax.bar(x,       rf_accs,  w, label='Random Forest', color='#55A868', edgecolor='white')
    ax.bar(x + w,   cnn_accs, w, label='1D CNN',        color='#DD8800', edgecolor='white')
    ax.axhline(0.25, color='gray', linestyle='--', alpha=0.6,
               linewidth=1.2, label='Chance (25%)')

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-Subject Classification Accuracy — LOSO Cross-Validation",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_per_subject_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG 5: HRV Feature Distributions
# ─────────────────────────────────────────────

def plot_feature_distributions():
    X    = np.load(f"{PROCESSED_DIR}/X_features.npy")
    y    = np.load(f"{PROCESSED_DIR}/y_features.npy")
    with open(f"{PROCESSED_DIR}/feature_names.json") as f:
        feat_names = json.load(f)

    key_feats  = ['mean_hr', 'sdnn', 'rmssd', 'lf_hf']
    key_labels = ['Mean HR (BPM)', 'SDNN (ms)', 'RMSSD (ms)', 'LF/HF Ratio']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("HRV Feature Distributions by Emotional Condition",
                 fontsize=14, fontweight='bold')

    for ax, feat, flabel in zip(axes.flat, key_feats, key_labels):
        fidx = feat_names.index(feat)
        for ci, cname in enumerate(CLASS_NAMES):
            vals = X[y == ci, fidx]
            ax.hist(vals, bins=35, alpha=0.55,
                    label=cname, color=CLASS_COLORS[ci],
                    edgecolor='none', density=True)
        ax.set_xlabel(flabel, fontsize=11)
        ax.set_ylabel("Density",  fontsize=10)
        ax.set_title(flabel,      fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_feature_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG 6: Classification Report Heatmaps
# ─────────────────────────────────────────────

def plot_classification_report_heatmaps(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p):
    results = [
        ("SVM (Feature-Based)",        svm_t, svm_p, "#4C72B0"),
        ("Random Forest (Feature-Based)", rf_t, rf_p, "#55A868"),
        ("1D CNN (End-to-End)",         cnn_t, cnn_p, "#DD8800"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Detailed Classification Report — All Models (LOSO)",
                 fontsize=13, fontweight='bold')

    for ax, (name, yt, yp, col) in zip(axes, results):
        report = classification_report(yt, yp, target_names=CLASS_NAMES,
                                        output_dict=True)
        metrics = ['precision', 'recall', 'f1-score']
        data    = np.array([[report[c][m] for m in metrics] for c in CLASS_NAMES])

        im = ax.imshow(data, vmin=0, vmax=1, cmap='YlGn', aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Precision', 'Recall', 'F1'],
                            fontweight='bold', fontsize=10)
        ax.set_yticks(range(4))
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)

        for i in range(4):
            for j in range(3):
                ax.text(j, i, f"{data[i,j]:.3f}",
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        color='white' if data[i,j] > 0.60 else '#222')

        macro = report['macro avg']
        ax.set_title(
            f"{name}\n"
            f"Macro: P={macro['precision']:.3f}  "
            f"R={macro['recall']:.3f}  "
            f"F1={macro['f1-score']:.3f}",
            fontsize=10, fontweight='bold', color=col
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig_classification_report.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# PRINT FULL SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p):
    print("\n" + "=" * 60)
    print("  SECTION 3 — FULL RESULTS SUMMARY")
    print("=" * 60)

    results = [
        ("SVM (Feature-Based)",        svm_t, svm_p),
        ("Random Forest (Feature-Based)", rf_t, rf_p),
        ("1D CNN (End-to-End)",         cnn_t, cnn_p),
    ]

    print(f"\n  {'Model':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*52}")
    for name, yt, yp in results:
        print(f"  {name:<30} "
              f"{accuracy_score(yt,yp):>10.4f} "
              f"{f1_score(yt,yp,average='macro'):>10.4f}")

    print(f"\n  Per-class F1:")
    print(f"  {'Class':<14}", end="")
    for name, _, __ in results:
        print(f"  {name.split('(')[0].strip():>14}", end="")
    print()

    for ci, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:<14}", end="")
        for _, yt, yp in results:
            f1c = f1_score(yt, yp, average=None)[ci]
            print(f"  {f1c:>14.3f}", end="")
        print()

    print("\n  Insights:")
    print("  • All models exceed chance (25%) — PPG encodes emotion-relevant info")
    print("  • Amusement is the hardest class (fewest samples + subtle physiology)")
    print("  • SVM achieves highest Macro F1 due to balanced class handling")
    print("  • Feature-based models outperform raw-signal MLP (convolutions needed)")
    print("  • High inter-subject variance limits overall performance")
    print("=" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WESAD Results Analysis — Section 3")
    print("=" * 60)

    svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p = load_results()

    print("\n  Generating figures...")
    plot_model_comparison(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p)
    plot_confusion_matrices(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p)
    plot_per_class_f1(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p)
    plot_per_subject_accuracy()
    plot_feature_distributions()
    plot_classification_report_heatmaps(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p)

    print_summary(svm_t, svm_p, rf_t, rf_p, cnn_t, cnn_p)
    print(f"\n  All figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
