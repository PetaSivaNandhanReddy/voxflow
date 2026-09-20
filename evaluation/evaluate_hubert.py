#!/usr/bin/env python3
"""
VoxFlow V2 — Model 2: HuBERT Locked TEST Set Evaluation Script
Evaluates the trained HuBERT multi-label classifier on the untouched TEST set (6,812 clips)
using the optimal thresholds selected exclusively on the VALIDATION set.

Outputs:
  - v2/evaluation/model2_hubert_test_metrics.json
  - v2/evaluation/MODEL2_HUBERT_TEST_REPORT.md

Reports metrics both on the full TEST set and on the strictly assigned-speaker subset (quarantining the 156 panel clips).
"""

import os
import sys
import io
import time
import json
import wave
import zipfile
import argparse
import pathlib
import numpy as np
import pandas as pd

# Lazy imports for torch / transformers
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, hamming_loss, accuracy_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import dataset and model from train_hubert
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training"))
try:
    from train_hubert import StutteringAudioDataset, HubertMultiLabelClassifier
except ImportError:
    pass


def evaluate_split(y_true, y_probs, thresholds, class_names):
    """
    Computes all standard multi-label metrics using frozen thresholds.
    """
    class_metrics = {}
    binary_preds = np.zeros_like(y_probs, dtype=int)

    for i, name in enumerate(class_names):
        y_t = y_true[:, i]
        y_p = y_probs[:, i]
        th = thresholds[name]

        preds = (y_p >= th).astype(int)
        binary_preds[:, i] = preds

        roc_auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.0

        f1 = float(f1_score(y_t, preds, zero_division=0))
        prec = float(precision_score(y_t, preds, zero_division=0))
        rec = float(recall_score(y_t, preds, zero_division=0))

        # Default 0.50 threshold comparison
        def_preds = (y_p >= 0.50).astype(int)
        def_f1 = float(f1_score(y_t, def_preds, zero_division=0))

        class_metrics[name] = {
            'positive_support': int(y_t.sum()),
            'positive_pct': round(float(y_t.sum() / len(y_t) * 100), 2),
            'threshold_used': th,
            'roc_auc': round(roc_auc, 4),
            'pr_auc': round(pr_auc, 4),
            'f1_score': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_default_0.5': round(def_f1, 4)
        }

    macro_f1 = np.mean([class_metrics[c]['f1_score'] for c in class_names])
    mean_pr_auc = np.mean([class_metrics[c]['pr_auc'] for c in class_names])
    mean_roc_auc = np.mean([class_metrics[c]['roc_auc'] for c in class_names])
    subset_acc = float(accuracy_score(y_true, binary_preds))
    h_loss = float(hamming_loss(y_true, binary_preds))

    return {
        'total_samples': len(y_true),
        'macro_f1': round(float(macro_f1), 4),
        'mean_pr_auc': round(float(mean_pr_auc), 4),
        'mean_roc_auc': round(float(mean_roc_auc), 4),
        'subset_accuracy': round(subset_acc, 4),
        'hamming_loss': round(h_loss, 4),
        'class_metrics': class_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="VoxFlow V2 HuBERT TEST Set Evaluation")
    
    def _resolve(rel):
        p1 = pathlib.Path(rel)
        if p1.exists():
            return str(p1)
        p2 = PROJECT_ROOT / rel
        if p2.exists():
            return str(p2)
        return str(PROJECT_ROOT / rel)

    def_ckpt = _resolve('models/Hubert_D/hubert_stage_B_best.pt')
    def_val = _resolve('models/Hubert_D/val_summary.json')
    def_test = _resolve('dataset/manifests/v2_test_manifest.csv')
    def_zip = _resolve('dataset/raw/sep28k_preextracted.zip')
    def_out = str(PROJECT_ROOT / 'evaluation')

    parser.add_argument('--checkpoint_path', type=str, default=def_ckpt)
    parser.add_argument('--val_summary_path', type=str, default=def_val)
    parser.add_argument('--test_manifest', type=str, default=def_test)
    parser.add_argument('--zip_path', type=str, default=def_zip)
    parser.add_argument('--audio_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader workers (default: 0 for Mac/MPS)")
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=def_out)
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required to run evaluation.")
        sys.exit(1)

    print("==================================================")
    print("VoxFlow V2 — Model 2: HuBERT Locked TEST Set Evaluation")
    print("==================================================")

    # 1. Load locked validation thresholds
    if os.path.exists(args.val_summary_path):
        with open(args.val_summary_path) as f:
            val_summary = json.load(f)
        locked_thresholds = val_summary['optimal_thresholds']
    elif os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, map_location='cpu')
        locked_thresholds = ckpt.get('optimal_thresholds', {'Repetition': 0.5, 'Prolongation': 0.5, 'Block': 0.5})
    else:
        raise FileNotFoundError(f"Neither {args.val_summary_path} nor {args.checkpoint_path} exists.")

    print(f"Locked Validation Thresholds: {locked_thresholds}")

    # Device selection (CUDA -> MPS -> CPU)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    if device.type == 'mps':
        print("  -> Apple Silicon MPS acceleration enabled.")

    # 2. Load Checkpoint & Model
    print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    pretrained_name = ckpt.get('args', {}).get('pretrained_model', 'facebook/hubert-base-ls960')

    model = HubertMultiLabelClassifier(pretrained_model=pretrained_name, num_labels=3, freeze_encoder=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # 3. Load TEST Dataset
    test_dataset = StutteringAudioDataset(args.test_manifest, zip_path=args.zip_path, audio_dir=args.audio_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Total TEST samples to evaluate: {len(test_dataset):,}")

    # 4. Generate Predictions
    all_probs = []
    all_targets = []
    print("Running inference on TEST set...")
    start_time = time.time()

    with torch.no_grad():
        for waveforms, targets in test_loader:
            waveforms = waveforms.to(device)
            logits = model(waveforms)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.1f}s ({len(test_dataset)/elapsed:.1f} clips/s)")

    y_true_all = np.vstack(all_targets)
    y_probs_all = np.vstack(all_probs)
    class_names = ['Repetition', 'Prolongation', 'Block']

    # 5. Evaluate Full TEST Set (6,812 clips)
    full_eval = evaluate_split(y_true_all, y_probs_all, locked_thresholds, class_names)

    # 6. Evaluate Assigned-Speaker TEST Subset (6,656 clips, excluding 156 panel clips)
    test_df = pd.read_csv(args.test_manifest)
    assigned_mask = (test_df['speaker'] != 'UNASSIGNED_MULTI_SPEAKER').values
    if assigned_mask.sum() < len(test_df):
        assigned_eval = evaluate_split(y_true_all[assigned_mask], y_probs_all[assigned_mask], locked_thresholds, class_names)
        panel_count = len(test_df) - int(assigned_mask.sum())
    else:
        assigned_eval = full_eval
        panel_count = 0

    print("\n==================================================")
    print("Model 2 (HuBERT) Full TEST Set Results (Locked Validation Thresholds)")
    print("==================================================")
    print(f"Macro F1:        {full_eval['macro_f1']:.4f}")
    print(f"Mean PR-AUC:     {full_eval['mean_pr_auc']:.4f}")
    print(f"Mean ROC-AUC:    {full_eval['mean_roc_auc']:.4f}")
    print(f"Subset Accuracy: {full_eval['subset_accuracy']:.4f}")
    print(f"Hamming Loss:    {full_eval['hamming_loss']:.4f}")
    print("--------------------------------------------------")
    for cn in class_names:
        m = full_eval['class_metrics'][cn]
        print(f"{cn:12s} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | F1: {m['f1_score']:.4f} (Th: {m['threshold_used']}) | P: {m['precision']:.4f} | R: {m['recall']:.4f}")
    print("==================================================")

    # 7. Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "model2_hubert_test_metrics.json")
    md_path = os.path.join(args.out_dir, "MODEL2_HUBERT_TEST_REPORT.md")

    final_metrics_payload = {
        'model': 'Model 2: Pretrained HuBERT Multi-Label Classifier',
        'pretrained_checkpoint': pretrained_name,
        'checkpoint_evaluated': args.checkpoint_path,
        'evaluation_protocol': 'LOCKED TEST EVALUATION (Thresholds chosen on Validation Only)',
        'locked_thresholds': locked_thresholds,
        'full_test_set': full_eval,
        'assigned_speaker_test_subset': assigned_eval,
        'quarantined_panel_clips_in_test': panel_count
    }

    with open(json_path, 'w') as f:
        json.dump(final_metrics_payload, f, indent=2)
    print(f"Saved evaluation metrics to {json_path}")

    # Write Markdown report
    with open(md_path, 'w') as f:
        f.write("# VoxFlow V2 — Model 2: HuBERT TEST Set Evaluation Report\n\n")
        f.write(f"**Pretrained Checkpoint:** `{pretrained_name}`  \n")
        f.write(f"**Checkpoint Evaluated:** `{args.checkpoint_path}`  \n")
        f.write(f"**Total TEST Samples:** {full_eval['total_samples']:,} clips  \n")
        f.write(f"**Quarantined Panel Clips:** {panel_count} clips  \n\n")
        f.write("## 1. Full TEST Set Metrics\n\n")
        f.write(f"- **Macro F1:** **{full_eval['macro_f1']:.4f}**\n")
        f.write(f"- **Mean PR-AUC:** **{full_eval['mean_pr_auc']:.4f}**\n")
        f.write(f"- **Mean ROC-AUC:** **{full_eval['mean_roc_auc']:.4f}**\n")
        f.write(f"- **Subset Accuracy:** **{full_eval['subset_accuracy']:.4f}**\n")
        f.write(f"- **Hamming Loss:** **{full_eval['hamming_loss']:.4f}**\n\n")
        f.write("| Class | Positive Support | Prevalence | Locked Threshold | ROC-AUC | PR-AUC | Precision | Recall | Optimal F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for cn in class_names:
            m = full_eval['class_metrics'][cn]
            f.write(f"| **{cn}** | {m['positive_support']:,} | {m['positive_pct']}% | {m['threshold_used']} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1_score']:.4f}** |\n")
        f.write("\n---\n\n")
        f.write("## 2. Strictly Assigned-Speaker TEST Subset (Panel Clips Quarantined)\n\n")
        f.write(f"- **Samples:** {assigned_eval['total_samples']:,} clips  \n")
        f.write(f"- **Macro F1:** **{assigned_eval['macro_f1']:.4f}**  \n")
        f.write(f"- **Mean PR-AUC:** **{assigned_eval['mean_pr_auc']:.4f}**  \n")
        f.write(f"- **Mean ROC-AUC:** **{assigned_eval['mean_roc_auc']:.4f}**  \n\n")
        f.write("| Class | Positive Support | Prevalence | Locked Threshold | ROC-AUC | PR-AUC | Precision | Recall | Optimal F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for cn in class_names:
            m = assigned_eval['class_metrics'][cn]
            f.write(f"| **{cn}** | {m['positive_support']:,} | {m['positive_pct']}% | {m['threshold_used']} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1_score']:.4f}** |\n")

    print(f"Saved evaluation report to {md_path}")


if __name__ == '__main__':
    main()
