# VoxFlow V2 — Model 3: Wav2Vec2 Main Model (Kaggle GPU Training Guide)

This guide documents how to train **Model 3: Wav2Vec2 Main Model (`facebook/wav2vec2-base`)** on Kaggle GPU (Tesla T4).

---

## 1. Files in this Kaggle Package

Upload the following files to `/kaggle/working/`:
- `train_wav2vec2.py` — Standalone PyTorch/Transformers training script
- `v2_train_manifest.csv` — Locked training split (17,555 clips, 18 speakers)
- `v2_val_manifest.csv` — Locked validation split (6,632 clips, 126 speakers)

*(The TEST set manifest `v2_test_manifest.csv` is strictly untouched and not included).*

---

## 2. Input Audio Setup

Mount the extracted audio dataset in your Kaggle Notebook sidebar:
- Example path: `/kaggle/input/datasets/nayanojwalsritej/voxflow-sep28k-audio`
- The script automatically resolves: `audio_dir / archive_path`
- ZIP is **not** required when `--audio_dir` is provided.

---

## 3. Environment Setup (Notebook Cell 1)

Use the verified isolated environment:

```bash
%%bash
python3 -m venv --system-site-packages /kaggle/working/voxflow_env
/kaggle/working/voxflow_env/bin/pip install --quiet --no-cache-dir "transformers==5.17.0" || \
/kaggle/working/voxflow_env/bin/pip install --quiet --no-cache-dir "transformers==4.57.6"

# Verify environment in clean subprocess
/kaggle/working/voxflow_env/bin/python -c "
import torch, transformers
from transformers import Wav2Vec2Model, Wav2Vec2Config
print('CUDA Available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
print('--> Wav2Vec2 environment ready!')
"
```

---

## 4. Stage A — Frozen Backbone Baseline (Notebook Cell 2)

Trains the 3-output multi-label classification head while keeping the entire Wav2Vec2 backbone frozen:

```bash
/kaggle/working/voxflow_env/bin/python /kaggle/working/train_wav2vec2.py \
  --stage A \
  --pretrained_model facebook/wav2vec2-base \
  --train_manifest /kaggle/working/v2_train_manifest.csv \
  --val_manifest /kaggle/working/v2_val_manifest.csv \
  --audio_dir /kaggle/input/datasets/nayanojwalsritej/voxflow-sep28k-audio \
  --output_dir /kaggle/working/wav2vec2_stageA \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --lr 1e-3 \
  --epochs 5 \
  --fp16 \
  --num_workers 0
```

### Stage A Outputs (in `/kaggle/working/wav2vec2_stageA/`):
- `wav2vec2_stage_A_best.pt`
- `wav2vec2_stage_A_val_summary.json`
- `wav2vec2_stage_A_history.json`
- `wav2vec2_stage_A_config.json`

---

## 5. Stage B — Controlled Fine-Tuning (Notebook Cell 3)

Starts from `wav2vec2_stage_A_best.pt`, unfreezes only the final 2 transformer encoder layers (layers 10 and 11), and fine-tunes with differential learning rates:

```bash
/kaggle/working/voxflow_env/bin/python /kaggle/working/train_wav2vec2.py \
  --stage B \
  --pretrained_model facebook/wav2vec2-base \
  --checkpoint_in /kaggle/working/wav2vec2_stageA/wav2vec2_stage_A_best.pt \
  --train_manifest /kaggle/working/v2_train_manifest.csv \
  --val_manifest /kaggle/working/v2_val_manifest.csv \
  --audio_dir /kaggle/input/datasets/nayanojwalsritej/voxflow-sep28k-audio \
  --output_dir /kaggle/working/wav2vec2_stageB \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --lr_encoder 1e-5 \
  --lr_head 1e-4 \
  --unfreeze_last_n_layers 2 \
  --epochs 3 \
  --fp16 \
  --num_workers 0
```

### Stage B Outputs (in `/kaggle/working/wav2vec2_stageB/`):
- `wav2vec2_stage_B_best.pt`
- `wav2vec2_stage_B_val_summary.json`
- `wav2vec2_stage_B_history.json`
- `wav2vec2_stage_B_config.json`

---

## 6. Stage C — Deeper Fine-Tuning (Notebook Cell 4)

Starts from `wav2vec2_stageB_extended/wav2vec2_stage_B_best.pt`, unfreezes the final 4 transformer encoder layers (layers 8, 9, 10, 11), and trains for 5 epochs with conservative encoder learning rate (`5e-6`):

```bash
/kaggle/working/voxflow_env/bin/python /kaggle/working/train_wav2vec2.py \
  --stage C \
  --pretrained_model facebook/wav2vec2-base \
  --checkpoint_in /kaggle/working/wav2vec2_stageB_extended/wav2vec2_stage_B_best.pt \
  --train_manifest /kaggle/working/v2_train_manifest.csv \
  --val_manifest /kaggle/working/v2_val_manifest.csv \
  --audio_dir /kaggle/input/datasets/nayanojwalsritej/voxflow-sep28k-audio \
  --output_dir /kaggle/working/wav2vec2_stageC \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --lr_encoder 5e-6 \
  --lr_head 1e-4 \
  --unfreeze_last_n_layers 4 \
  --epochs 5 \
  --fp16 \
  --num_workers 0
```

### Stage C Outputs (in `/kaggle/working/wav2vec2_stageC/`):
- `wav2vec2_stage_C_best.pt`
- `wav2vec2_stage_C_val_summary.json`
- `wav2vec2_stage_C_history.json`
- `wav2vec2_stage_C_config.json`

