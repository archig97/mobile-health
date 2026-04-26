"""
=============================================================
WESAD Course Project — Step 2b: Classical ML Classifiers
CSC 491/591 Ubiquitous Computing and Mobile Health
=============================================================
Approach 2 (Feature-Based):
  - SVM  : Radial Basis Function kernel, C=10, class-weighted
  - Random Forest : 200 trees, max depth 15, class-weighted

Evaluation:
  Leave-One-Subject-Out (LOSO) Cross-Validation
  → Each fold: train on 14 subjects, test on the left-out subject
  → Prevents data leakage between subjects (windows from the same
    subject are correlated, so random splits would inflate results)

Metrics: Accuracy, Macro F1, per-class F1, Confusion Matrix

Key design choices:
  - StandardScaler fit ONLY on training fold (re-fit each fold)
    to prevent information leakage from test subject
  - class_weight='balanced' compensates for Amusement underrepresentation
  - Hyperparameters selected based on literature + grid search intuition
=============================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PROCESSED_DIR = "./processed"
OUTPUT_DIR    = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES  = ["Baseline", "Stress", "Amusement", "Meditation"]
LABEL_NAMES  = {1: "Baseline", 2: "Stress", 3: "Amusement", 4: "Meditation"}
CLASS_COLORS = ["#4C72B0", "#DD4949", "#55A868", "#8172B2"]

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def make_svm():
    """
    SVM with RBF kernel.
    - C=10        : moderate regularization; penalizes misclassification
    - gamma=scale : 1/(n_features * X.var()) — good default for normalized data
    - class_weight='balanced': upweights minority classes (Amusement)
    """
    return SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    )


def make_xgboost():
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )


def make_random_forest():
    """
    Random Forest ensemble.
    - n_estimators=200 : 200 trees (diminishing returns above ~200)
    - max_depth=15     : prevents overfitting on 12-feature input
    - class_weight='balanced_subsample': rebalance per bootstrap sample
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1          # use all CPU cores
    )


# ─────────────────────────────────────────────
# LOSO CROSS-VALIDATION
# ─────────────────────────────────────────────

def loso_cv(X, y, subjects, make_clf, clf_name, verbose=True):
    """
    Leave-One-Subject-Out cross-validation.

    For each subject S:
      train = all windows NOT from S
      test  = all windows FROM S
      → fit scaler on train only (no leakage)
      → fit classifier on train
      → predict on test

    Returns: (all_true, all_pred) — arrays concatenated across all folds
    """
    unique_subjects = sorted(np.unique(subjects))
    all_true, all_pred = [], []
    per_subj_results   = {}

    if verbose:
        print(f"\n  {clf_name} — LOSO ({len(unique_subjects)} folds)")
        print(f"  {'Subject':<8} {'N_test':>7} {'Acc':>7} {'Macro F1':>10}")
        print(f"  {'-'*36}")

    for test_subj in unique_subjects:
        train_mask = subjects != test_subj
        test_mask  = subjects == test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # Scale: fit ONLY on training data
        scaler   = StandardScaler()
        X_train  = scaler.fit_transform(X_train)
        X_test   = scaler.transform(X_test)

        # Oversample minority classes in training fold only (never touch test)
        min_class_count = int(np.bincount(y_train)[np.unique(y_train)].min())
        try:
            smote   = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except ValueError:
            ros     = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)

        # Train and predict
        clf   = make_clf()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        all_true.extend(y_test)
        all_pred.extend(preds)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='macro')
        per_subj_results[test_subj] = {'acc': acc, 'f1': f1, 'n': int(test_mask.sum())}

        if verbose:
            print(f"  {test_subj:<8} {test_mask.sum():>7} {acc:>7.3f} {f1:>10.3f}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1  = f1_score(all_true, all_pred, average='macro')

    if verbose:
        print(f"  {'-'*36}")
        print(f"  {'OVERALL':<8} {len(all_true):>7} {overall_acc:>7.3f} {overall_f1:>10.3f}")

    return all_true, all_pred, per_subj_results


# ─────────────────────────────────────────────
# RESULTS & FIGURES
# ─────────────────────────────────────────────

def print_report(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"  {name} — Classification Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred,
                                 target_names=CLASS_NAMES, digits=3))


def plot_confusion_matrix(y_true, y_pred, title, filepath):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Row Proportion', 'shrink': 0.85},
                ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Label",      fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12,  fontweight='bold', pad=10)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_feature_importance(rf_model, feat_names, filepath):
    importances = rf_model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#55A868" if i < 3 else "#AACFAA" for i in range(len(feat_names))]
    ax.bar(range(len(feat_names)),
           importances[sorted_idx],
           color=colors, edgecolor='white')
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels([feat_names[i] for i in sorted_idx],
                        rotation=35, ha='right', fontsize=10)
    ax.set_ylabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Random Forest — Feature Importance",
                 fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WESAD Classical Classifiers — LOSO Cross-Validation")
    print("=" * 60)

    # Load features
    X    = np.load(f"{PROCESSED_DIR}/X_features.npy")
    y    = np.load(f"{PROCESSED_DIR}/y_features.npy")
    subj = np.load(f"{PROCESSED_DIR}/subj_features.npy")

    with open(f"{PROCESSED_DIR}/feature_names.json") as f:
        feat_names = json.load(f)

    print(f"\n  Features: {X.shape}  (windows × features)")
    print(f"  Labels  : {np.unique(y)} → {[LABEL_NAMES[i] for i in np.unique(y)]}")
    print(f"  Subjects: {sorted(np.unique(subj))}")

    # ── SVM ──
    svm_true, svm_pred, svm_subj = loso_cv(
        X, y, subj, make_svm, "SVM (RBF, C=10)")
    print_report("SVM", svm_true, svm_pred)
    plot_confusion_matrix(
        svm_true, svm_pred,
        f"SVM — Confusion Matrix (LOSO)\nAcc={accuracy_score(svm_true,svm_pred):.3f}  "
        f"Macro F1={f1_score(svm_true,svm_pred,average='macro'):.3f}",
        f"{OUTPUT_DIR}/cm_svm.png")

    # ── Random Forest ──
    rf_true, rf_pred, rf_subj = loso_cv(
        X, y, subj, make_random_forest, "Random Forest (200 trees)")
    print_report("Random Forest", rf_true, rf_pred)
    plot_confusion_matrix(
        rf_true, rf_pred,
        f"Random Forest — Confusion Matrix (LOSO)\nAcc={accuracy_score(rf_true,rf_pred):.3f}  "
        f"Macro F1={f1_score(rf_true,rf_pred,average='macro'):.3f}",
        f"{OUTPUT_DIR}/cm_rf.png")

    # ── XGBoost ── (requires 0-indexed labels)
    y_xgb = y - 1
    xgb_true, xgb_pred, xgb_subj = loso_cv(
        X, y_xgb, subj, make_xgboost, "XGBoost (200 trees)")
    print_report("XGBoost", xgb_true, xgb_pred)
    plot_confusion_matrix(
        xgb_true, xgb_pred,
        f"XGBoost — Confusion Matrix (LOSO)\nAcc={accuracy_score(xgb_true,xgb_pred):.3f}  "
        f"Macro F1={f1_score(xgb_true,xgb_pred,average='macro'):.3f}",
        f"{OUTPUT_DIR}/cm_xgb.png")

    # Feature importance (train on ALL data for visualization only)
    print("\n  Computing feature importance (full dataset)...")
    scaler_full = StandardScaler()
    X_scaled    = scaler_full.fit_transform(X)
    rf_full     = make_random_forest()
    rf_full.fit(X_scaled, y)
    plot_feature_importance(rf_full, feat_names,
                            f"{OUTPUT_DIR}/feature_importance.png")

    # ── Save results ──
    results = {
        "svm": {
            "accuracy":       float(accuracy_score(svm_true, svm_pred)),
            "macro_f1":       float(f1_score(svm_true, svm_pred, average='macro')),
            "per_class_f1":   f1_score(svm_true, svm_pred, average=None).tolist(),
            "per_subject":    svm_subj,
        },
        "rf": {
            "accuracy":       float(accuracy_score(rf_true, rf_pred)),
            "macro_f1":       float(f1_score(rf_true, rf_pred, average='macro')),
            "per_class_f1":   f1_score(rf_true, rf_pred, average=None).tolist(),
            "per_subject":    rf_subj,
        },
        "feature_names": feat_names,
        "feature_importance": rf_full.feature_importances_.tolist(),
        "class_names": CLASS_NAMES,
    }

    # Convert per_subject dicts (already dicts, not arrays)
    results["svm"]["per_subject"]  = svm_subj
    results["rf"]["per_subject"]   = rf_subj

    np.save(f"{OUTPUT_DIR}/svm_true.npy", svm_true)
    np.save(f"{OUTPUT_DIR}/svm_pred.npy", svm_pred)
    np.save(f"{OUTPUT_DIR}/rf_true.npy",  rf_true)
    np.save(f"{OUTPUT_DIR}/rf_pred.npy",  rf_pred)
    np.save(f"{OUTPUT_DIR}/xgb_true.npy", xgb_true)
    np.save(f"{OUTPUT_DIR}/xgb_pred.npy", xgb_pred)

    with open(f"{OUTPUT_DIR}/classical_results.json", "w") as f:
        # Convert non-serializable types
        r = {
            "svm": {"accuracy": results["svm"]["accuracy"],
                    "macro_f1": results["svm"]["macro_f1"],
                    "per_class_f1": results["svm"]["per_class_f1"],
                    "per_subject": {k: {"acc": v["acc"], "f1": v["f1"]}
                                    for k, v in svm_subj.items()}},
            "rf":  {"accuracy": results["rf"]["accuracy"],
                    "macro_f1": results["rf"]["macro_f1"],
                    "per_class_f1": results["rf"]["per_class_f1"],
                    "per_subject": {k: {"acc": v["acc"], "f1": v["f1"]}
                                    for k, v in rf_subj.items()}},
            "xgb": {"accuracy": float(accuracy_score(xgb_true, xgb_pred)),
                    "macro_f1": float(f1_score(xgb_true, xgb_pred, average='macro')),
                    "per_class_f1": f1_score(xgb_true, xgb_pred, average=None).tolist(),
                    "per_subject": {k: {"acc": v["acc"], "f1": v["f1"]}
                                    for k, v in xgb_subj.items()}},
        }
        json.dump(r, f, indent=2)

    print(f"\n  Results saved to {OUTPUT_DIR}/")
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*47}")
    print(f"  {'SVM (Feature-Based)':<25} "
          f"{accuracy_score(svm_true,svm_pred):>10.4f} "
          f"{f1_score(svm_true,svm_pred,average='macro'):>10.4f}")
    print(f"  {'Random Forest (Feature)':<25} "
          f"{accuracy_score(rf_true,rf_pred):>10.4f} "
          f"{f1_score(rf_true,rf_pred,average='macro'):>10.4f}")
    print(f"  {'XGBoost (Feature)':<25} "
          f"{accuracy_score(xgb_true,xgb_pred):>10.4f} "
          f"{f1_score(xgb_true,xgb_pred,average='macro'):>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
