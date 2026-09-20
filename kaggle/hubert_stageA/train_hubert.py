#!/usr/bin/env python3
"""
VoxFlow V2 — Model 2: HuBERT Competitor Training Script
Pretrained HuBERT Multi-Label Stuttering Event Classifier
Targets: Repetition, Prolongation, Block (3 independent binary outputs)

Supports:
  - STAGE A: Frozen encoder (pre-trained HuBERT representations evaluated directly)
  - STAGE B: Limited fine-tuning (conservative learning rate + early stopping)

Execution:
  Designed for standalone execution on Kaggle GPU or local environments.
  On-the-fly streaming of 3-second 16kHz mono PCM WAV audio from ZIP or extracted folder.
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

# Lazy imports / conditional imports for torch/transformers
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    try:
        from torch.amp import autocast, GradScaler
    except ImportError:
        from torch.cuda.amp import autocast, GradScaler
    from transformers import HubertModel, HubertConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ==============================================================================
# Audio Dataset Definition
# ==============================================================================
class StutteringAudioDataset(Dataset if TORCH_AVAILABLE else object):
    """
    On-the-fly audio loader for 16kHz mono 3-second clips.
    Reads directly from ZIP archive or extracted audio directory to conserve RAM.
    Thread-safe and multi-process safe for PyTorch DataLoader workers.
    """
    def __init__(self, manifest_path, zip_path=None, audio_dir=None, target_samples=48000):
        self.df = pd.read_csv(manifest_path)
        self.zip_path = zip_path if (zip_path and os.path.exists(zip_path)) else None
        self.audio_dir = audio_dir if (audio_dir and os.path.exists(audio_dir)) else None
        self.target_samples = target_samples
        self._zip_handle = None
        self._zip_pid = None
        self._zip_namelist_map = None

        if not self.zip_path and not self.audio_dir:
            raise ValueError(f"Neither valid zip_path ('{zip_path}') nor audio_dir ('{audio_dir}') provided.")

        self.labels = self.df[['repetition_label', 'prolongation_label', 'block_label']].values.astype(np.float32)
        self.archive_paths = self.df['archive_path'].values

    def _get_zip(self):
        current_pid = os.getpid()
        if self._zip_handle is None or self._zip_pid != current_pid:
            self._zip_handle = zipfile.ZipFile(self.zip_path, 'r')
            self._zip_pid = current_pid
        return self._zip_handle

    def __len__(self):
        return len(self.df)

    def _read_wav_bytes(self, raw_bytes):
        with wave.open(io.BytesIO(raw_bytes), 'rb') as wf:
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Pad or trim to exactly target_samples (48,000 for 3.0s at 16kHz)
        if len(audio) < self.target_samples:
            pad = np.zeros(self.target_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])
        elif len(audio) > self.target_samples:
            audio = audio[:self.target_samples]
        return audio

    def __getitem__(self, idx):
        rel_path = self.archive_paths[idx]
        raw_bytes = None

        if self.audio_dir:
            full_path = os.path.join(self.audio_dir, rel_path)
            if not os.path.exists(full_path):
                # Fallback to basename
                full_path = os.path.join(self.audio_dir, os.path.basename(rel_path))
            with open(full_path, 'rb') as f:
                raw_bytes = f.read()
        elif self.zip_path:
            zh = self._get_zip()
            try:
                raw_bytes = zh.read(rel_path)
            except KeyError:
                if self._zip_namelist_map is None:
                    self._zip_namelist_map = {os.path.basename(n): n for n in zh.namelist()}
                basename = os.path.basename(rel_path)
                if basename in self._zip_namelist_map:
                    raw_bytes = zh.read(self._zip_namelist_map[basename])
                else:
                    raise KeyError(f"Clip '{rel_path}' not found in archive '{self.zip_path}'")

        waveform = self._read_wav_bytes(raw_bytes)
        label = self.labels[idx]

        if TORCH_AVAILABLE:
            return torch.from_numpy(waveform), torch.from_numpy(label)
        return waveform, label


# ==============================================================================
# Model Definition
# ==============================================================================
if TORCH_AVAILABLE:
    class HubertMultiLabelClassifier(nn.Module):
        """
        HuBERT-base sequence classifier for multi-label stuttering detection.
        Temporal representation is mean-pooled and passed through a lightweight MLP.
        """
        def __init__(self, pretrained_model="facebook/hubert-base-ls960", num_labels=3, freeze_encoder=True):
            super().__init__()
            self.hubert = HubertModel.from_pretrained(pretrained_model)
            self.hidden_size = self.hubert.config.hidden_size  # 768
            self.num_labels = num_labels
            self.freeze_encoder = freeze_encoder

            # Freeze or unfreeze HuBERT encoder
            self.set_encoder_freeze(freeze_encoder)

            # Lightweight classification head
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_labels)
            )

        def set_encoder_freeze(self, freeze=True):
            self.freeze_encoder = freeze
            for param in self.hubert.parameters():
                param.requires_grad = not freeze
            if freeze:
                self.hubert.eval()
            else:
                self.hubert.train()

        def set_stage_b_freeze(self, unfreeze_last_n_layers=2):
            """
            Stage B fine-tuning: unfreezes only the last n transformer layers.
            All earlier encoder layers and the CNN feature extractor remain strictly frozen.
            """
            self.freeze_encoder = False
            self.unfreeze_last_n_layers = unfreeze_last_n_layers

            # 1. Freeze ALL HuBERT parameters first
            for param in self.hubert.parameters():
                param.requires_grad = False

            # 2. Unfreeze only the final n transformer layers
            if hasattr(self.hubert, 'encoder') and hasattr(self.hubert.encoder, 'layers'):
                layers = self.hubert.encoder.layers
                total_layers = len(layers)
                start_idx = max(0, total_layers - unfreeze_last_n_layers)
                for idx in range(start_idx, total_layers):
                    for param in layers[idx].parameters():
                        param.requires_grad = True

            # 3. Ensure classification head is trainable
            for param in self.classifier.parameters():
                param.requires_grad = True

        def train(self, mode=True):
            super().train(mode)
            if self.freeze_encoder:
                # Stage A: keep entire backbone in eval mode
                self.hubert.eval()
            else:
                # Stage B: keep CNN feature extractor & earlier layers in eval mode
                self.hubert.eval()
                if hasattr(self.hubert, 'encoder') and hasattr(self.hubert.encoder, 'layers'):
                    unfrozen_n = getattr(self, 'unfreeze_last_n_layers', 2)
                    for layer in self.hubert.encoder.layers[-unfrozen_n:]:
                        layer.train(mode)
                self.classifier.train(mode)
            return self

        def forward(self, input_values):
            if self.freeze_encoder:
                with torch.no_grad():
                    outputs = self.hubert(input_values=input_values)
                    pooled = torch.mean(outputs.last_hidden_state, dim=1)
            else:
                outputs = self.hubert(input_values=input_values)
                pooled = torch.mean(outputs.last_hidden_state, dim=1)
            logits = self.classifier(pooled)
            return logits
else:
    class HubertMultiLabelClassifier(object):
        pass


# ==============================================================================
# Threshold Optimization (Validation Only)
# ==============================================================================
def optimize_thresholds_on_val(y_true, y_probs, class_names):
    """
    Performs grid-search on validation set only to find optimal F1 thresholds.
    Range: 0.05 to 0.95, step: 0.01.
    """
    threshold_grid = np.arange(0.05, 0.96, 0.01)
    results = {}

    for i, name in enumerate(class_names):
        y_t = y_true[:, i]
        y_p = y_probs[:, i]

        roc_auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.0

        best_th = 0.50
        best_f1 = -1.0
        best_prec = 0.0
        best_rec = 0.0

        for th in threshold_grid:
            preds = (y_p >= th).astype(int)
            f1 = f1_score(y_t, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(round(th, 2))
                best_prec = float(precision_score(y_t, preds, zero_division=0))
                best_rec = float(recall_score(y_t, preds, zero_division=0))

        # Default 0.50 threshold comparison
        default_preds = (y_p >= 0.50).astype(int)
        default_f1 = float(f1_score(y_t, default_preds, zero_division=0))

        results[name] = {
            'auc_roc': round(roc_auc, 4),
            'pr_auc': round(pr_auc, 4),
            'optimal_threshold': best_th,
            'val_f1_optimal': round(best_f1, 4),
            'val_precision': round(best_prec, 4),
            'val_recall': round(best_rec, 4),
            'val_f1_default_0.5': round(default_f1, 4)
        }

    return results


# ==============================================================================
# Validation Evaluation Function
# ==============================================================================
def evaluate_validation(model, val_loader, device, criterion, class_names):
    model.eval()
    total_val_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for waveforms, targets in val_loader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            logits = model(waveforms)
            loss = criterion(logits, targets)
            total_val_loss += loss.item() * waveforms.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    val_loss = total_val_loss / len(val_loader.dataset)
    y_true = np.vstack(all_targets)
    y_probs = np.vstack(all_probs)

    class_metrics = optimize_thresholds_on_val(y_true, y_probs, class_names)
    macro_f1 = np.mean([class_metrics[c]['val_f1_optimal'] for c in class_names])
    mean_pr_auc = np.mean([class_metrics[c]['pr_auc'] for c in class_names])
    mean_roc_auc = np.mean([class_metrics[c]['auc_roc'] for c in class_names])

    summary = {
        'val_loss': round(val_loss, 4),
        'macro_f1': round(float(macro_f1), 4),
        'mean_pr_auc': round(float(mean_pr_auc), 4),
        'mean_roc_auc': round(float(mean_roc_auc), 4),
        'class_metrics': class_metrics
    }
    return summary


# ==============================================================================
# Main Training Routine
# ==============================================================================
def train(args):
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch and Transformers must be installed to run training.")
        sys.exit(1)

    print("==================================================")
    print(f"VoxFlow V2 — HuBERT Multi-Label Training (Stage {args.stage})")
    print("==================================================")
    print(f"Pretrained Model: {args.pretrained_model}")
    print(f"Stage:            {'STAGE A: Frozen HuBERT Backbone' if args.stage == 'A' else 'STAGE B: Limited Fine-Tuning'}")
    print(f"Batch Size:       {args.batch_size} (Grad Accum: {args.grad_accum_steps})")
    print(f"Learning Rate:    {args.lr}")
    print(f"Epochs:           {args.epochs}")
    print(f"Random Seed:      {args.seed}")
    print("==================================================")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    elif device.type == 'cuda':
        print(f"  -> NVIDIA CUDA acceleration enabled: {torch.cuda.get_device_name(0)}")
    else:
        print("  -> Running on CPU.")

    # Datasets and Loaders
    train_dataset = StutteringAudioDataset(args.train_manifest, zip_path=args.zip_path, audio_dir=args.audio_dir)
    val_dataset = StutteringAudioDataset(args.val_manifest, zip_path=args.zip_path, audio_dir=args.audio_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    # Calculate exact positive weights from TRAIN ONLY
    train_labels = train_dataset.labels
    num_train = len(train_labels)
    num_pos = train_labels.sum(axis=0)
    pos_weights = (num_train - num_pos) / num_pos
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    class_names = ['Repetition', 'Prolongation', 'Block']
    print("\nClass Imbalance Weights (TRAIN ONLY):")
    for name, pw, p_count in zip(class_names, pos_weights, num_pos):
        print(f"  - {name:12s}: positive support = {int(p_count):5d} ({p_count/num_train*100:5.2f}%), pos_weight = {pw:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Initialize model
    freeze_encoder = (args.stage == 'A')
    model = HubertMultiLabelClassifier(
        pretrained_model=args.pretrained_model,
        num_labels=3,
        freeze_encoder=freeze_encoder
    )

    if args.checkpoint_in and os.path.exists(args.checkpoint_in):
        print(f"Loading weights from checkpoint: {args.checkpoint_in}")
        ckpt = torch.load(args.checkpoint_in, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print("  -> Successfully loaded Stage A checkpoint into model.")

    if args.stage == 'B':
        model.set_stage_b_freeze(unfreeze_last_n_layers=args.unfreeze_last_n_layers)

    model.to(device)

    # Confirm layer trainability and print parameter counts
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    if args.stage == 'B':
        print("\n==================================================")
        print("Stage B Layer Trainability Verification")
        print("==================================================")
        trainable_layer_indices = set()
        frozen_layer_indices = set()
        other_trainable = []
        other_frozen = []

        for name, p in model.named_parameters():
            if 'encoder.layers.' in name:
                layer_idx = int(name.split('encoder.layers.')[1].split('.')[0])
                if p.requires_grad:
                    trainable_layer_indices.add(layer_idx)
                else:
                    frozen_layer_indices.add(layer_idx)
            else:
                comp = name.split('.')[0] + '.' + name.split('.')[1] if '.' in name else name
                if p.requires_grad:
                    other_trainable.append(comp)
                else:
                    other_frozen.append(comp)

        other_trainable = sorted(list(set(other_trainable)))
        other_frozen = sorted(list(set(other_frozen)))

        print("HuBERT Transformer Layers (0-11):")
        print(f"  - Frozen layers (0-9):        {sorted(list(frozen_layer_indices))}")
        print(f"  - Unfrozen layers (10-11):    {sorted(list(trainable_layer_indices))}")
        print("Non-encoder components:")
        print(f"  - Frozen non-target parts:    {other_frozen}")
        print(f"  - Trainable head parts:       {other_trainable}")
        print("Parameter Counts:")
        print(f"  - Trainable parameters:       {trainable_params_count:,}")
        print(f"  - Frozen parameters:          {frozen_params_count:,}")
        print(f"CONFIRMATION: Only encoder layers 10-11 and classifier head are trainable: {trainable_layer_indices == {10, 11} and all('classifier' in c for c in other_trainable)}")
        print("==================================================\n")

    # Optimizer configuration
    if args.stage == 'B':
        encoder_params = [p for p in model.hubert.parameters() if p.requires_grad]
        classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]

        optimizer_grouped_parameters = [
            {'params': encoder_params, 'lr': args.lr_encoder, 'weight_decay': args.weight_decay},
            {'params': classifier_params, 'lr': args.lr_head, 'weight_decay': args.weight_decay}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        print(f"Optimizer Parameter Groups (Stage B AdamW):")
        print(f"  - HuBERT Unfrozen Encoder (layers 10-11): {sum(p.numel() for p in encoder_params):,} params | lr = {args.lr_encoder}")
        print(f"  - Classification Head:                  {sum(p.numel() for p in classifier_params):,} params | lr = {args.lr_head}")
        print(f"  - Weight Decay:                         {args.weight_decay}")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"\nOptimizer (Stage A AdamW):")
        print(f"  - Trainable Params (Head): {sum(p.numel() for p in trainable_params):,} | lr = {args.lr}")
        print(f"  - Frozen Params (HuBERT):  {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    use_cuda_amp = (args.fp16 and device.type == 'cuda')
    scaler = GradScaler(enabled=use_cuda_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_f1 = -1.0
    best_summary = None
    best_checkpoint_path = os.path.join(args.output_dir, f"hubert_stage_{args.stage}_best.pt")
    training_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()

        optimizer.zero_grad()
        for step, (waveforms, targets) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            if use_cuda_amp:
                with autocast(device_type='cuda', enabled=True):
                    logits = model(waveforms)
                    loss = criterion(logits, targets)
                    loss = loss / args.grad_accum_steps
            else:
                logits = model(waveforms)
                loss = criterion(logits, targets)
                loss = loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * args.grad_accum_steps * waveforms.size(0)

            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch}/{args.epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item() * args.grad_accum_steps:.4f}")

        train_loss = total_train_loss / len(train_dataset)
        elapsed = time.time() - start_time

        # Validation evaluation
        print(f"\nEvaluating on Validation Set (Epoch {epoch})...")
        val_summary = evaluate_validation(model, val_loader, device, criterion, class_names)
        print(f"Epoch {epoch} finished in {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_summary['val_loss']:.4f}")
        print(f"Val Macro F1: {val_summary['macro_f1']:.4f} | Mean PR-AUC: {val_summary['mean_pr_auc']:.4f} | Mean ROC-AUC: {val_summary['mean_roc_auc']:.4f}")

        # Record history entry
        training_history.append({
            'epoch': epoch,
            'train_loss': round(float(train_loss), 4),
            'val_loss': val_summary['val_loss'],
            'val_macro_f1': val_summary['macro_f1'],
            'val_mean_pr_auc': val_summary['mean_pr_auc'],
            'val_mean_roc_auc': val_summary['mean_roc_auc'],
            'class_metrics': val_summary['class_metrics'],
            'elapsed_seconds': round(float(elapsed), 1)
        })

        # Check for best validation score
        if val_summary['macro_f1'] > best_val_f1:
            best_val_f1 = val_summary['macro_f1']
            best_summary = val_summary
            torch.save({
                'epoch': epoch,
                'stage': args.stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_val_f1,
                'val_metrics': val_summary,
                'optimal_thresholds': {c: val_summary['class_metrics'][c]['optimal_threshold'] for c in class_names},
                'pos_weights': pos_weights.tolist(),
                'args': vars(args)
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path}")

    # 1. Save final validation summary JSON
    summary_path = os.path.join(args.output_dir, f"hubert_stage_{args.stage}_val_summary.json")
    final_output = {
        'model_architecture': f"HuBERT Multi-Label Classifier ({args.pretrained_model})",
        'stage': args.stage,
        'encoder_frozen': (args.stage == 'A'),
        'unfrozen_encoder_layers': (args.unfreeze_last_n_layers if args.stage == 'B' else 0),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': 6812,
        'test_status': 'UNTOUCHED',
        'metrics_validation': best_summary['class_metrics'],
        'mean_metrics': {
            'macro_f1': best_summary['macro_f1'],
            'mean_pr_auc': best_summary['mean_pr_auc'],
            'mean_roc_auc': best_summary['mean_roc_auc']
        },
        'optimal_thresholds': {c: best_summary['class_metrics'][c]['optimal_threshold'] for c in class_names},
        'checkpoint_path': best_checkpoint_path,
        'stage_a_val_macro_f1': 0.4508
    }
    with open(summary_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"\nFinal validation metrics saved to {summary_path}")

    # 2. Save training history JSON
    history_path = os.path.join(args.output_dir, f"hubert_stage_{args.stage}_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # 3. Save run configuration JSON
    config_path = os.path.join(args.output_dir, f"hubert_stage_{args.stage}_config.json")
    run_config = {
        'pretrained_model': args.pretrained_model,
        'stage': args.stage,
        'encoder_frozen': (args.stage == 'A'),
        'unfrozen_encoder_layers': (args.unfreeze_last_n_layers if args.stage == 'B' else 0),
        'random_seed': args.seed,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum_steps,
        'effective_batch_size': args.batch_size * args.grad_accum_steps,
        'learning_rate': (args.lr if args.stage == 'A' else {'encoder': args.lr_encoder, 'head': args.lr_head}),
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'fp16_enabled': (args.fp16 and device.type == 'cuda'),
        'device_used': str(device),
        'pos_weights': {name: float(pw) for name, pw in zip(class_names, pos_weights)},
        'class_names': class_names,
        'checkpoint_in': args.checkpoint_in,
        'test_set_touched': False
    }
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"Run configuration saved to {config_path}")

    # 4. Print comprehensive summary report
    print("\n==================================================")
    print(f"HuBERT Stage {args.stage} Final Validation Report")
    print("==================================================")
    print(f"Best Validation Macro F1: {best_summary['macro_f1']:.4f}")
    if args.stage == 'B':
        diff = best_summary['macro_f1'] - 0.4508
        print(f"Stage A Baseline Macro F1: 0.4508")
        print(f"Delta (Stage B vs Stage A): {diff:+.4f} ({'+' if diff >= 0 else ''}{diff/0.4508*100:.2f}%)")
    print(f"\nPer-Class Validation Results (Optimal Thresholds):")
    for c in class_names:
        m = best_summary['class_metrics'][c]
        print(f"  - {c:12s}: F1={m['val_f1_optimal']:.4f} (Prec={m['val_precision']:.4f}, Rec={m['val_recall']:.4f}) | ROC-AUC={m['auc_roc']:.4f} | PR-AUC={m['pr_auc']:.4f} | Th={m['optimal_threshold']:.2f}")
    print("\nDefault (0.50 Threshold) Comparison:")
    for c in class_names:
        m = best_summary['class_metrics'][c]
        print(f"  - {c:12s}: Default 0.50 F1 = {m['val_f1_default_0.5']:.4f}")
    print("\nLocked TEST set remains STRICTLY untouched.")
    print("==================================================")


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def _resolve_default_file(*rel_paths):
    for rp in rel_paths:
        p1 = pathlib.Path(rp)
        if p1.exists():
            return str(p1)
        p2 = PROJECT_ROOT / rp
        if p2.exists():
            return str(p2)
    return str(PROJECT_ROOT / rel_paths[0])

def parse_args():
    parser = argparse.ArgumentParser(description="VoxFlow V2 HuBERT Multi-Label Training")
    parser.add_argument('--stage', type=str, default='A', choices=['A', 'B'], help="Training stage: A (frozen) or B (fine-tuning)")
    parser.add_argument('--pretrained_model', type=str, default='facebook/hubert-base-ls960', help="Pretrained checkpoint name")
    
    default_train = _resolve_default_file('v2_train_manifest.csv', 'dataset/manifests/v2_train_manifest.csv')
    default_val = _resolve_default_file('v2_val_manifest.csv', 'dataset/manifests/v2_val_manifest.csv')
    default_zip = _resolve_default_file('v2/dataset/raw/sep28k_preextracted.zip', 'dataset/raw/sep28k_preextracted.zip')
    default_out = 'output' if os.path.exists('v2_train_manifest.csv') else str(PROJECT_ROOT / 'models/hubert_stageA')
    
    parser.add_argument('--train_manifest', type=str, default=default_train, help="Path to train manifest CSV")
    parser.add_argument('--val_manifest', type=str, default=default_val, help="Path to validation manifest CSV")
    parser.add_argument('--zip_path', type=str, default=default_zip, help="Path to sep28k_preextracted.zip")
    parser.add_argument('--audio_dir', type=str, default=None, help="Directory containing audio if unzipped")
    parser.add_argument('--output_dir', type=str, default=default_out, help="Output directory for checkpoints and metrics")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per device (default: 16 for Stage A, 8 for Stage B)")
    parser.add_argument('--grad_accum_steps', type=int, default=2, help="Gradient accumulation steps (effective batch size = 32)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Stage A learning rate (default: 1e-3)")
    parser.add_argument('--lr_encoder', type=float, default=1e-5, help="Stage B encoder learning rate (default: 1e-5)")
    parser.add_argument('--lr_head', type=float, default=1e-4, help="Stage B classification head learning rate (default: 1e-4)")
    parser.add_argument('--unfreeze_last_n_layers', type=int, default=2, help="Number of final HuBERT transformer layers to unfreeze in Stage B (default: 2)")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs (default: 5 for Stage A, 3 for Stage B)")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader workers (default: 0 for Apple Silicon MPS and Mac stability)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true', help="Use mixed precision training (CUDA only)")
    parser.add_argument('--device', type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu')")
    parser.add_argument('--checkpoint_in', type=str, default=None, help="Input checkpoint path for Stage B (e.g. hubert_stage_A_best.pt)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
