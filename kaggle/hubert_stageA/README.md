# VoxFlow V2 — HuBERT Stage A (Kaggle GPU Training Guide)

This guide explains how to run **Model 2: HuBERT Competitor (Stage A — Frozen Encoder)** on Kaggle GPU using the self-contained package.

---

## 1. What Files to Upload to Kaggle

Upload only the **4 files** inside `v2/kaggle/hubert_stageA/` (~3.9 MB total):

1. `train_hubert.py` — Standalone PyTorch/Transformers training script
2. `v2_train_manifest.csv` — Locked training split (17,555 clips, 18 speakers)
3. `v2_val_manifest.csv` — Locked validation split (6,632 clips, 126 speakers)
4. `hubert_config.json` — Hyperparameter and class weighting configuration

*(Do NOT upload the full VoxFlow repository, Model 1 caches, or TEST manifests).*

---

## 2. How the Audio Dataset Should be Attached

Attach your existing pre-extracted audio archive:
- **File:** `sep28k_preextracted.zip` (2.33 GB)
- **Kaggle Setup:** In your Kaggle Notebook sidebar, click **Add Input** -> **Upload Dataset** -> Select your local `v2/dataset/raw/sep28k_preextracted.zip`.
- Once mounted, Kaggle places it at `/kaggle/input/<dataset-name>/sep28k_preextracted.zip`.

---

## 3. Expected Kaggle Directory Structure

Inside your notebook environment:

```text
/kaggle/
├── input/
│   └── <dataset-name>/
│       └── sep28k_preextracted.zip       <-- 2.33 GB attached audio dataset
└── working/
    ├── train_hubert.py                   <-- Uploaded from package
    ├── v2_train_manifest.csv             <-- Uploaded from package
    ├── v2_val_manifest.csv               <-- Uploaded from package
    ├── hubert_config.json                <-- Uploaded from package
    └── output/                           <-- Created automatically during training
        ├── hubert_stage_A_best.pt        <-- Best model checkpoint (~380 MB)
        ├── hubert_stage_A_val_summary.json
        ├── hubert_stage_A_history.json
        └── hubert_stage_A_config.json
```

---

## 4. Isolated Environment & Package Setup (Notebook Cell 1)

Kaggle's base environment can experience import conflicts with default `transformers` packages. To ensure 100% stability, create a lightweight isolated virtual environment in `/kaggle/working/` that reuses Kaggle's preinstalled CUDA PyTorch while cleanly installing the compatible Transformers release:

```bash
%%bash
python3 -m venv --system-site-packages /kaggle/working/voxflow_env
/kaggle/working/voxflow_env/bin/pip install --quiet --no-cache-dir "transformers==5.17.0" || \
/kaggle/working/voxflow_env/bin/pip install --quiet --no-cache-dir "transformers==4.57.6"

# Verify in a clean subprocess
/kaggle/working/voxflow_env/bin/python -c "
import torch, transformers
from transformers import HubertModel, HubertForSequenceClassification
print('CUDA available:', torch.cuda.is_available())
print('Transformers:', transformers.__version__)
print('--> HuBERT imports verified successfully!')
"
```

---

## 5. Exact Stage A Execution Command (Notebook Cell 2)

Run the standalone training script using the isolated environment's Python executable. The script streams audio on-the-fly directly from the ZIP file, eliminating RAM overflow:

```bash
/kaggle/working/voxflow_env/bin/python train_hubert.py \
  --stage A \
  --pretrained_model facebook/hubert-base-ls960 \
  --train_manifest v2_train_manifest.csv \
  --val_manifest v2_val_manifest.csv \
  --zip_path /kaggle/input/<dataset-name>/sep28k_preextracted.zip \
  --output_dir /kaggle/working/output \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --lr 0.001 \
  --epochs 5 \
  --fp16
```

*(Note: Replace `<dataset-name>` with your actual Kaggle dataset mount directory).*

---

## 6. Training Behavior & Output Artifacts

### What Happens During Stage A:
1. **Backbone Frozen:** Loads `facebook/hubert-base-ls960` and freezes all 12 transformer encoder layers (`requires_grad = False`).
2. **Trainable Head:** Trains a lightweight classifier (`Linear(768, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 3)`).
3. **Class Imbalance:** Uses `BCEWithLogitsLoss` with positive weights derived strictly from TRAIN:
   - **Repetition:** `4.4149` (18.47% positive)
   - **Prolongation:** `9.3143` (9.70% positive)
   - **Block:** `7.8483` (11.30% positive)
4. **Validation Only:** After each epoch, sweeps thresholds ($0.05$ to $0.95$, step $0.01$) on `v2_val_manifest.csv` to select the threshold that maximizes per-class F1.
5. **TEST Untouched:** The TEST manifest is **not** loaded or inspected.

### Generated Artifacts in `/kaggle/working/output/`:
- `hubert_stage_A_best.pt` (~380 MB): PyTorch model checkpoint.
- `hubert_stage_A_val_summary.json`: Detailed per-class metrics and locked thresholds.
- `hubert_stage_A_history.json`: Epoch-by-epoch loss, timing, and metric progression.
- `hubert_stage_A_config.json`: Recorded runtime configuration and seed metadata.

---

## 7. How to Copy Artifacts Back to Local VoxFlow

After training finishes on Kaggle:
1. Download the generated files from `/kaggle/working/output/`.
2. Copy them into your local VoxFlow workspace at:
   ```text
   v2/models/hubert_stageA/
   ├── hubert_stage_A_best.pt
   ├── hubert_stage_A_val_summary.json
   ├── hubert_stage_A_history.json
   └── hubert_stage_A_config.json
   ```

---

## 8. What Output to Paste Back for Analysis

Once training completes on Kaggle, copy and paste:
1. The **terminal output log** (especially epoch loss values, training speed, and validation summaries).
2. The complete text of **`hubert_stage_A_val_summary.json`**.
