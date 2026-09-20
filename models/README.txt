
VOXFLOW V2 — OBJECTIVE 1 MODEL ARTIFACTS
=========================================

SVM Baseline
------------
svm = Multi-Label LinearSVC (Calibrated)
- val_summary.json

HuBERT
------
Hubert_A = HuBERT Stage A (Frozen Encoder)
Hubert_B = HuBERT Stage B (Unfrozen Layers 10-11, 3 Epochs)
Hubert_C = HuBERT Stage B Extended (Unfrozen Layers 10-11, 5 Epochs)
Hubert_D = HuBERT Stage B Extended v2 (Selected Application Model)
- config.json
- history.json
- val_summary.json
- hubert_stage_B_best.pt (Canonical Checkpoint for Objective 2)

Wav2Vec2
--------
Wav2Vec2_A = Wav2Vec2 Stage A
Wav2Vec2_B = Wav2Vec2 Stage B
Wav2Vec2_C = Wav2Vec2 Stage B Extended
Wav2Vec2_D = Wav2Vec2 Stage C
Wav2Vec2_E = Wav2Vec2 Mean+Max Stage C
- config.json
- history.json
- val_summary.json

Final test results are locked and stored under:
v2/results/
- hubert_test_results.json
- hubert_test_per_class.csv
- wav2vec2_test_results.json
- wav2vec2_test_per_class.csv

Final Selected Application Model:
Hubert_D (Macro F1 = 0.4599, PR-AUC = 0.4532, ROC-AUC = 0.8099)
