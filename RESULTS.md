# VoxFlow — Model Benchmark Results

This document presents the experimental results for speech disfluency detection evaluated on the locked VoxFlow dataset (30,999 clips total).

All models were evaluated under strict speaker-exclusive splitting:
- **Train split:** 17,555 clips (18 speakers)
- **Validation split:** 6,632 clips (126 speakers) — used for threshold optimization
- **Locked Test split:** 6,812 clips (137 unseen speakers) — final evaluation

---

## 1. Overall Test Set Benchmark Comparison

| Model | Architecture / Backbone | Val Macro F1 | Test Macro F1 | Test Mean PR-AUC | Test Mean ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Multi-label LinearSVC (162 MFCC features) | 0.3262 | 0.3121 | 0.2545 | 0.6633 | Baseline |
| **Wav2Vec2** | Fine-tuned `facebook/wav2vec2-base` | 0.4479 | 0.4498 | 0.4288 | 0.8020 | Evaluated |
| **HuBERT** | Fine-tuned `facebook/hubert-base-ls960` | **0.4718** | **0.4599** | **0.4532** | **0.8099** | **Selected Production Model** |

---

## 2. Production Model (HuBERT) Per-Class Performance

Evaluated on 6,812 unseen-speaker test clips using validation-optimized decision thresholds:

| Disfluency Target | Decision Threshold | Test Precision | Test Recall | Test F1 | Test PR-AUC | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Repetition** | **0.77** | 0.5201 | 0.5983 | **0.5565** | 0.5971 | 0.8465 |
| **Prolongation** | **0.79** | 0.5096 | 0.4611 | **0.4841** | 0.4844 | 0.8550 |
| **Block** | **0.57** | 0.2558 | 0.5035 | **0.3393** | 0.2781 | 0.7281 |
| **Mean / Macro** | — | 0.4285 | 0.5210 | **0.4599** | **0.4532** | **0.8099** |

- **Subset Accuracy (Exact Multi-label Match):** 57.93%
- **Hamming Loss:** 0.1709

---

## 3. Comparison Summary

- **HuBERT** achieved the highest overall Macro F1 (**0.4599**), Mean PR-AUC (**0.4532**), and Mean ROC-AUC (**0.8099**) across all disfluency classes on held-out speakers.
- Pre-trained self-supervised acoustic representations substantially outperformed traditional acoustic feature engineering (MFCC + SVM Macro F1: 0.3121).
- HuBERT was selected as the sole active inference model in the VoxFlow production application.
