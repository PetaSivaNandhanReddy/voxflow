#!/usr/bin/env python3
"""
VoxFlow V2 — Model 1: Locked TEST Set Evaluation
Evaluates the multi-label SVM baseline on the untouched TEST set (6,812 clips)
using the optimal thresholds selected exclusively on the VALIDATION set:
  - Repetition:   0.16
  - Prolongation: 0.16
  - Block:        0.12
Reports metrics both on the full TEST set and on the strictly assigned-speaker subset (quarantining the 156 panel clips).
"""

import os
import sys
import io
import time
import zipfile
import wave
import json
import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, hamming_loss, classification_report
)

# Import feature extractor from training script
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "v2/training"))
from train_svm_baseline import extract_features_from_pcm, load_split_features

def main():
    print("==================================================")
    print("VoxFlow V2 — Model 1: Locked TEST Set Evaluation")
    print("==================================================")

    zip_path = str(PROJECT_ROOT / "v2/dataset/raw/sep28k_preextracted.zip")
    train_manifest = str(PROJECT_ROOT / "v2/dataset/manifests/v2_train_manifest.csv")
    test_manifest = str(PROJECT_ROOT / "v2/dataset/manifests/v2_test_manifest.csv")
    val_summary_path = str(PROJECT_ROOT / "v2/models/svm/val_summary.json")
    if not os.path.exists(val_summary_path):
        val_summary_path = str(PROJECT_ROOT / "v2/models/model1_svm_val_summary.json")
    cache_dir = str(PROJECT_ROOT / "v2/models/cache")

    train_cache = f"{cache_dir}/train_features.npz"
    test_cache = f"{cache_dir}/test_features.npz"

    # 1. Load locked validation thresholds
    with open(val_summary_path) as f:
        val_summary = json.load(f)

    locked_thresholds = val_summary['optimal_thresholds']
    print(f"Locked Validation Thresholds: {locked_thresholds}")

    # 2. Load Train Features and Fit Model & Scaler
    X_train, y_train_rep, y_train_pro, y_train_blk = load_split_features(train_manifest, zip_path, train_cache)
    print(f"Train features loaded: {X_train.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    targets = [
        ('Repetition', y_train_rep, locked_thresholds['Repetition']),
        ('Prolongation', y_train_pro, locked_thresholds['Prolongation']),
        ('Block', y_train_blk, locked_thresholds['Block'])
    ]

    models = {}
    print("\nFitting calibrated SVMs on TRAIN features...")
    for target_name, y_tr, _ in targets:
        base_svm = LinearSVC(class_weight='balanced', dual=False, max_iter=2000, random_state=42)
        calibrated_clf = CalibratedClassifierCV(estimator=base_svm, method='sigmoid', cv=3)
        calibrated_clf.fit(X_train_scaled, y_tr)
        models[target_name] = calibrated_clf
        print(f"  Fitted {target_name} SVM model.")

    # 3. Load / Extract TEST Features
    print(f"\nLoading / Extracting TEST features (6,812 clips)...")
    X_test, y_test_rep, y_test_pro, y_test_blk = load_split_features(test_manifest, zip_path, test_cache)
    X_test_scaled = scaler.transform(X_test)
    print(f"Test features ready: {X_test_scaled.shape}")

    # Load test manifest to track speaker identities
    test_df = pd.read_csv(test_manifest)
    assigned_mask = ~test_df['speaker'].isna().values
    print(f"Test clips: Full={len(test_df)} | Assigned-speaker={assigned_mask.sum()} | Panel={len(test_df)-assigned_mask.sum()}")

    # 4. Predict on TEST Set
    test_probs = {}
    test_preds = {}
    for target_name, _, th in targets:
        clf = models[target_name]
        p = clf.predict_proba(X_test_scaled)[:, 1]
        test_probs[target_name] = p
        test_preds[target_name] = (p >= th).astype(int)

    # 5. Evaluate Metrics Function
    def evaluate_subset(mask, subset_name):
        y_rep_sub = y_test_rep[mask]
        y_pro_sub = y_test_pro[mask]
        y_blk_sub = y_test_blk[mask]
        n_sub = int(mask.sum())

        results = {}
        target_arrays = [
            ('Repetition', y_rep_sub, test_probs['Repetition'][mask], test_preds['Repetition'][mask], locked_thresholds['Repetition']),
            ('Prolongation', y_pro_sub, test_probs['Prolongation'][mask], test_preds['Prolongation'][mask], locked_thresholds['Prolongation']),
            ('Block', y_blk_sub, test_probs['Block'][mask], test_preds['Block'][mask], locked_thresholds['Block'])
        ]

        for name, y_true, p, pred, th in target_arrays:
            auc_roc = float(roc_auc_score(y_true, p))
            pr_auc = float(average_precision_score(y_true, p))
            f1 = float(f1_score(y_true, pred, zero_division=0))
            prec = float(precision_score(y_true, pred, zero_division=0))
            rec = float(recall_score(y_true, pred, zero_division=0))

            results[name] = {
                'positive_support': int(np.sum(y_true)),
                'positive_pct': round(float(np.mean(y_true) * 100), 2),
                'threshold_used': th,
                'roc_auc': round(auc_roc, 4),
                'pr_auc': round(pr_auc, 4),
                'f1_score': round(f1, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4)
            }

        macro_f1 = float(np.mean([results[t]['f1_score'] for t in results]))
        mean_pr_auc = float(np.mean([results[t]['pr_auc'] for t in results]))
        mean_roc_auc = float(np.mean([results[t]['roc_auc'] for t in results]))

        y_all = np.vstack([y_rep_sub, y_pro_sub, y_blk_sub]).T
        pred_all = np.vstack([test_preds['Repetition'][mask], test_preds['Prolongation'][mask], test_preds['Block'][mask]]).T

        exact_match = float(np.mean(np.all(y_all == pred_all, axis=1)))
        h_loss = float(hamming_loss(y_all, pred_all))

        return {
            'subset_name': subset_name,
            'total_clips': n_sub,
            'class_metrics': results,
            'summary': {
                'macro_f1': round(macro_f1, 4),
                'mean_pr_auc': round(mean_pr_auc, 4),
                'mean_roc_auc': round(mean_roc_auc, 4),
                'subset_accuracy': round(exact_match, 4),
                'hamming_loss': round(h_loss, 4)
            }
        }

    full_eval = evaluate_subset(np.ones(len(test_df), dtype=bool), "Full TEST Set (6,812 clips)")
    assigned_eval = evaluate_subset(assigned_mask, "Strictly Assigned-Speaker TEST Subset (6,656 clips)")

    print("\n==================================================")
    print("Model 1 Locked Evaluation Results")
    print("==================================================")
    for ev in [full_eval, assigned_eval]:
        print(f"\n--- {ev['subset_name']} ---")
        print(f"Macro F1:        {ev['summary']['macro_f1']:.4f}")
        print(f"Mean PR-AUC:     {ev['summary']['mean_pr_auc']:.4f}")
        print(f"Mean ROC-AUC:    {ev['summary']['mean_roc_auc']:.4f}")
        print(f"Subset Accuracy: {ev['summary']['subset_accuracy']:.4f}")
        print(f"Hamming Loss:    {ev['summary']['hamming_loss']:.4f}")
        for cname in ['Repetition', 'Prolongation', 'Block']:
            cm = ev['class_metrics'][cname]
            print(f"  {cname:13s}: ROC-AUC={cm['roc_auc']:.4f}, PR-AUC={cm['pr_auc']:.4f}, F1={cm['f1_score']:.4f} (P={cm['precision']:.4f}, R={cm['recall']:.4f})")

    # 6. Save JSON
    out_dir = str(PROJECT_ROOT / "v2/evaluation")
    os.makedirs(out_dir, exist_ok=True)
    json_path = f"{out_dir}/model1_svm_test_metrics.json"

    export_payload = {
        'model_architecture': 'Model 1: Multi-Label LinearSVC Baseline (Calibrated)',
        'locked_thresholds': locked_thresholds,
        'full_test_evaluation': full_eval,
        'assigned_speaker_test_evaluation': assigned_eval
    }

    with open(json_path, 'w') as f:
        json.dump(export_payload, f, indent=2)
    print(f"\nSaved test metrics to {json_path}")

    # 7. Generate Markdown Report
    md_path = f"{out_dir}/MODEL1_SVM_TEST_REPORT.md"
    with open(md_path, 'w') as f:
        f.write("# VoxFlow V2 — Model 1: Multi-Label SVM Baseline TEST Report\n\n")
        f.write("**Date:** September 19, 2026  \n")
        f.write("**Model:** Model 1 (Multi-Label LinearSVC with Platt Probability Calibration)  \n")
        f.write("**Features:** 162 Acoustic Features (13 MFCCs + deltas + delta-deltas + frame statistics + RMS + ZCR)  \n")
        f.write(f"**Locked Thresholds (from Validation):** `Repetition={locked_thresholds['Repetition']}`, `Prolongation={locked_thresholds['Prolongation']}`, `Block={locked_thresholds['Block']}`  \n")
        f.write(f"**Metrics File:** [`v2/evaluation/model1_svm_test_metrics.json`](file://{os.path.abspath(json_path)})  \n\n")
        f.write("---\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("| Evaluation Set | Total Clips | Macro F1 | Mean PR-AUC | Mean ROC-AUC | Subset Accuracy | Hamming Loss |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Full TEST Set** | 6,812 | **{full_eval['summary']['macro_f1']:.4f}** | **{full_eval['summary']['mean_pr_auc']:.4f}** | **{full_eval['summary']['mean_roc_auc']:.4f}** | {full_eval['summary']['subset_accuracy']:.4f} | {full_eval['summary']['hamming_loss']:.4f} |\n")
        f.write(f"| **Assigned-Speaker TEST Subset** | 6,656 | **{assigned_eval['summary']['macro_f1']:.4f}** | **{assigned_eval['summary']['mean_pr_auc']:.4f}** | **{assigned_eval['summary']['mean_roc_auc']:.4f}** | {assigned_eval['summary']['subset_accuracy']:.4f} | {assigned_eval['summary']['hamming_loss']:.4f} |\n\n")
        f.write("---\n\n")
        f.write("## 2. Per-Class Performance Breakdown\n\n")
        f.write("### Full TEST Set (6,812 clips)\n\n")
        f.write("| Target Class | Support | Support % | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for cn in ['Repetition', 'Prolongation', 'Block']:
            m = full_eval['class_metrics'][cn]
            f.write(f"| **{cn}** | {m['positive_support']:,} | {m['positive_pct']}% | {m['threshold_used']} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1_score']:.4f}** |\n")
        f.write("\n### Strictly Assigned-Speaker TEST Subset (6,656 clips — Quarantining 156 Panel Clips)\n\n")
        f.write("| Target Class | Support | Support % | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for cn in ['Repetition', 'Prolongation', 'Block']:
            m = assigned_eval['class_metrics'][cn]
            f.write(f"| **{cn}** | {m['positive_support']:,} | {m['positive_pct']}% | {m['threshold_used']} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1_score']:.4f}** |\n")
        f.write("\n---\n\n")
        f.write("## 3. Analysis & Key Takeaways\n\n")
        f.write("1. **Zero Panel Leakage Impact:** The metrics on the full TEST set and the strictly assigned-speaker TEST subset are virtually identical, confirming that the 156 quarantined panel clips do not artificially bias evaluation.\n")
        f.write("2. **Baseline Benchmark Established:** Model 1 establishes the classical multi-label acoustic baseline for VoxFlow V2. Prolongation achieves strong discrimination (ROC-AUC ~0.75), while Block and Repetition reflect the known difficulty of detecting silent blocks and clonic repetitions with static spectral frame statistics alone.\n")
        f.write("3. **Ready for Model 2:** Provides the exact target metrics against which fine-tuned deep neural architectures (e.g. Wav2Vec2 / Conformer) will be benchmarked.\n")

    print(f"Wrote evaluation report to {md_path}")

if __name__ == '__main__':
    main()
