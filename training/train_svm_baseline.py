#!/usr/bin/env python3
"""
VoxFlow V2 — Model 1: Multi-Label SVM Baseline
Extracts 162 robust acoustic features (MFCC + delta + delta-delta + spectral + energy statistics),
trains 3 class-balanced linear SVMs on TRAIN (17,555 clips),
and performs optimal decision threshold selection on VALIDATION (6,632 clips).
TEST (6,812 clips) remains strictly untouched until final evaluation.
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
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report
)

# ---------------------------------------------------------
# Feature Extraction (Pure NumPy / SciPy, 0 external audio deps)
# ---------------------------------------------------------

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595.0) - 1)

def get_filter_banks(nfilt=40, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    lowmel = hz_to_mel(lowfreq)
    highmel = hz_to_mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    bin_points = np.floor((nfft + 1) * mel_to_hz(melpoints) / samplerate).astype(int)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])
    return fbank

# Precompute filterbank
FBANK = get_filter_banks(nfilt=40, nfft=512, samplerate=16000)

def compute_deltas(feat, N=2):
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N + 1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N + 1), padded[t : t + 2 * N + 1]) / denominator
    return delta_feat

def extract_features_from_pcm(raw_bytes, sample_rate=16000):
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    if len(audio) == 0:
        return np.zeros(162, dtype=np.float32)

    # Normalize to [-1.0, 1.0]
    audio = audio / 32768.0

    # 1. Pre-emphasis
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # 2. Framing (25ms frame, 10ms hop)
    frame_size = int(round(0.025 * sample_rate)) # 400 samples
    frame_stride = int(round(0.010 * sample_rate)) # 160 samples
    audio_len = len(emphasized)
    num_frames = max(1, int(np.ceil(float(np.abs(audio_len - frame_size)) / frame_stride)) + 1)

    pad_audio_len = (num_frames - 1) * frame_stride + frame_size
    pad_signal = np.pad(emphasized, (0, max(0, pad_audio_len - audio_len)), mode='constant')

    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_stride, frame_stride), (frame_size, 1)
    ).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 3. Hamming window
    frames *= np.hamming(frame_size)

    # 4. FFT & Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # 5. Mel Filterbank Energies
    filter_banks = np.dot(pow_frames, FBANK.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # 6. MFCCs (first 13 coefficients)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :13]

    # 7. Deltas and Delta-Deltas
    delta1 = compute_deltas(mfcc)
    delta2 = compute_deltas(delta1)

    all_mfcc = np.hstack([mfcc, delta1, delta2]) # Shape: (num_frames, 39)

    # Summary statistics across frames (39 * 4 = 156 features)
    mean_feat = np.mean(all_mfcc, axis=0)
    std_feat = np.std(all_mfcc, axis=0)
    min_feat = np.min(all_mfcc, axis=0)
    max_feat = np.max(all_mfcc, axis=0)

    # 8. Time-domain and spectral energy statistics (6 features)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

    rms_stats = np.array([np.mean(frame_rms), np.std(frame_rms), np.max(frame_rms)])
    zcr_stats = np.array([np.mean(zcr), np.std(zcr), np.max(zcr)])

    return np.concatenate([mean_feat, std_feat, min_feat, max_feat, rms_stats, zcr_stats]).astype(np.float32)

# ---------------------------------------------------------
# Feature Batch Loader directly from ZIP
# ---------------------------------------------------------

def load_split_features(manifest_path, zip_path, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        data = np.load(cache_path)
        return data['X'], data['y_rep'], data['y_pro'], data['y_blk']

    df = pd.read_csv(manifest_path)
    print(f"Extracting features for {len(df)} clips from {manifest_path}...")
    t0 = time.time()

    zf = zipfile.ZipFile(zip_path, 'r')
    X = np.zeros((len(df), 162), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        arch_path = row['archive_path']
        with zf.open(arch_path) as f:
            raw_bytes = f.read()
        # Parse WAV frames from memory
        with wave.open(io.BytesIO(raw_bytes), 'rb') as w:
            pcm_bytes = w.readframes(w.getnframes())
        X[i] = extract_features_from_pcm(pcm_bytes, row['sample_rate'])
        if (i + 1) % 2500 == 0 or (i + 1) == len(df):
            print(f"  Processed {i+1}/{len(df)} clips ({(time.time()-t0):.1f}s)...")

    y_rep = df['repetition_label'].values.astype(int)
    y_pro = df['prolongation_label'].values.astype(int)
    y_blk = df['block_label'].values.astype(int)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X, y_rep=y_rep, y_pro=y_pro, y_blk=y_blk)
        print(f"Saved feature cache to {cache_path}")

    return X, y_rep, y_pro, y_blk

# ---------------------------------------------------------
# Main Training & Validation Routine
# ---------------------------------------------------------

def main():
    print("==================================================")
    print("VoxFlow V2 — Model 1: Multi-Label SVM Baseline")
    print("==================================================")

    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    zip_path = str(project_root / "v2/dataset/raw/sep28k_preextracted.zip")
    train_manifest = str(project_root / "v2/dataset/manifests/v2_train_manifest.csv")
    val_manifest = str(project_root / "v2/dataset/manifests/v2_val_manifest.csv")
    cache_dir = str(project_root / "v2/models/cache")

    train_cache = f"{cache_dir}/train_features.npz"
    val_cache = f"{cache_dir}/val_features.npz"

    # 1. Feature Extraction
    X_train, y_train_rep, y_train_pro, y_train_blk = load_split_features(train_manifest, zip_path, train_cache)
    X_val, y_val_rep, y_val_pro, y_val_blk = load_split_features(val_manifest, zip_path, val_cache)

    print(f"\nFeature matrix shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")

    # 2. Standardization
    print("\nFitting StandardScaler on TRAIN features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    targets = [
        ('Repetition', y_train_rep, y_val_rep),
        ('Prolongation', y_train_pro, y_val_pro),
        ('Block', y_train_blk, y_val_blk)
    ]

    models = {}
    val_results = {}
    thresholds = {}

    print("\n==================================================")
    print("Training 3 Independent Class-Balanced SVMs")
    print("==================================================")

    for target_name, y_tr, y_va in targets:
        print(f"\n--- Target: {target_name.upper()} ---")
        pos_tr = np.sum(y_tr)
        pos_va = np.sum(y_va)
        print(f"Positive support: Train={pos_tr}/{len(y_tr)} ({pos_tr/len(y_tr)*100:.2f}%), Val={pos_va}/{len(y_va)} ({pos_va/len(y_va)*100:.2f}%)")

        # Class-balanced Linear SVM with probability calibration (Platt scaling via CalibratedClassifierCV)
        base_svm = LinearSVC(class_weight='balanced', dual=False, max_iter=2000, random_state=42)
        calibrated_clf = CalibratedClassifierCV(estimator=base_svm, method='sigmoid', cv=3)

        t_start = time.time()
        calibrated_clf.fit(X_train_scaled, y_tr)
        print(f"Fitted calibrated SVM in {time.time()-t_start:.2f}s")

        models[target_name] = calibrated_clf

        # Validation probabilities
        probs_val = calibrated_clf.predict_proba(X_val_scaled)[:, 1]
        auc_roc = roc_auc_score(y_va, probs_val)
        pr_auc = average_precision_score(y_va, probs_val)

        # Threshold sweep on validation set to optimize F1 score
        best_th = 0.5
        best_f1 = 0.0
        best_prec = 0.0
        best_rec = 0.0

        for th in np.arange(0.10, 0.90, 0.02):
            preds = (probs_val >= th).astype(int)
            f1 = f1_score(y_va, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                best_prec = precision_score(y_va, preds, zero_division=0)
                best_rec = recall_score(y_va, preds, zero_division=0)

        thresholds[target_name] = round(float(best_th), 2)
        val_results[target_name] = {
            'auc_roc': round(float(auc_roc), 4),
            'pr_auc': round(float(pr_auc), 4),
            'optimal_threshold': round(float(best_th), 2),
            'val_f1_optimal': round(float(best_f1), 4),
            'val_precision': round(float(best_prec), 4),
            'val_recall': round(float(best_rec), 4),
            'val_f1_default_0.5': round(float(f1_score(y_va, (probs_val >= 0.5).astype(int), zero_division=0)), 4)
        }

        print(f"ROC-AUC:           {auc_roc:.4f}")
        print(f"PR-AUC:            {pr_auc:.4f}")
        print(f"Optimal Threshold: {best_th:.2f} -> F1: {best_f1:.4f} (Prec: {best_prec:.4f}, Rec: {best_rec:.4f})")
        print(f"Default (0.50) F1: {val_results[target_name]['val_f1_default_0.5']:.4f}")

    # 3. Overall Multi-Label Evaluation on Validation Set
    val_preds_binary = np.zeros((len(X_val), 3), dtype=int)
    val_probs = np.zeros((len(X_val), 3), dtype=float)
    y_val_all = np.vstack([y_val_rep, y_val_pro, y_val_blk]).T

    for idx, (target_name, _, y_va) in enumerate(targets):
        clf = models[target_name]
        p = clf.predict_proba(X_val_scaled)[:, 1]
        val_probs[:, idx] = p
        val_preds_binary[:, idx] = (p >= thresholds[target_name]).astype(int)

    macro_f1 = np.mean([val_results[t]['val_f1_optimal'] for t in val_results])
    mean_pr_auc = np.mean([val_results[t]['pr_auc'] for t in val_results])
    mean_roc_auc = np.mean([val_results[t]['auc_roc'] for t in val_results])

    print("\n==================================================")
    print("Multi-Label Model 1 Validation Summary")
    print("==================================================")
    print(f"Mean ROC-AUC:   {mean_roc_auc:.4f}")
    print(f"Mean PR-AUC:    {mean_pr_auc:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Selected Thresholds: {thresholds}")
    print("TEST set remains STRICTLY untouched.")
    print("==================================================")

    # Save validation metrics to file
    out_dir = str(project_root / "v2/models")
    os.makedirs(out_dir, exist_ok=True)
    summary_data = {
        'model_architecture': 'Multi-Label LinearSVC (One-vs-Rest) with Calibrated Probabilities',
        'features': '162 Acoustic Features (13 MFCC + Delta + Delta-Delta + RMS + ZCR across frames)',
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': 6812,
        'test_status': 'UNTOUCHED',
        'metrics_validation': val_results,
        'mean_metrics': {
            'macro_f1': round(float(macro_f1), 4),
            'mean_pr_auc': round(float(mean_pr_auc), 4),
            'mean_roc_auc': round(float(mean_roc_auc), 4)
        },
        'optimal_thresholds': thresholds
    }

    with open(f"{out_dir}/model1_svm_val_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved validation metrics to {out_dir}/model1_svm_val_summary.json")

if __name__ == '__main__':
    main()
