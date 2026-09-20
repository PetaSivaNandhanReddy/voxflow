# VoxFlow V2 — Dataset Specification

This document summarizes the dataset sources, curation policies, and splits used in VoxFlow V2. Full quantitative distributions and verification hashes are recorded in `v2/audit/full_dataset_audit.json` and `v2/audit/v2_dataset_lock.json`.

---

## 1. Data Sources

VoxFlow V2 combines three official stuttering event datasets:
- **SEP-28k:** 28,177 audio clips from podcast episodes.
- **SEP-28k-Extended:** Extended speaker identities and podcast annotations.
- **FluencyBank:** 4,144 audio clips from adult stuttering interview sessions.
- **Combined Metadata Pool:** 32,321 clips (3.0-second, 16 kHz mono WAV audio).

---

## 2. Audio Verification & Usable Dataset Size

All 32,321 clips were verified against the pre-extracted audio archive (`sep28k_preextracted.zip`, 2.33 GB):
- **31,853 clips:** Full 3.0-second, 16 kHz, 16-bit PCM audio (exact size: 96,044 bytes).
- **413 clips:** 44-byte empty header files (timestamp超出 episode duration in original Apple extraction) — **excluded**.
- **55 clips:** Partial/truncated audio (< 3.0s duration) — **excluded**.

---

## 3. Clinical Quality Filtering

Clips with majority annotator flags indicating corrupt audio or lack of speech were excluded to prevent models from learning non-speech noise:
- `PoorAudioQuality >= 2` (577 clips excluded)
- `NoSpeech >= 2` (270 clips excluded)
- `Unsure >= 2` (38 clips excluded)
- **DifficultToUnderstand Preservation:** Clips marked `DifficultToUnderstand >= 2` were **retained** because clinical analysis revealed 466 of these clips contain severe stuttering events (226 blocks, 78 multi-label co-occurrences). Discarding them would artificially bias the model toward mild disfluencies.

**Final Clean Usable Dataset:** **30,999 clips**

---

## 4. Multi-Label Ground Truth Policy

VoxFlow V2 models **three independent stuttering disfluencies**:
1. **Repetition:** Sound repetitions (`SoundRep`) or word repetitions (`WordRep`).
2. **Prolongation:** Involuntary sound elongations (`Prolongation`).
3. **Block:** Silent postural fixations or articulatory blocks (`Block`).

### Decision Rule
A clip is assigned positive for a disfluency if **$\ge 2$ of 3 annotators** agree (majority consensus):
$$\text{Label}_c = 1 \iff \text{AnnotatorCount}_c \ge 2$$

- **Multi-Label Co-occurrence:** Clips can contain multiple disfluencies simultaneously (e.g., both Repetition and Block).
- **Downstream Fluent:** A clip is deemed fluent if all three labels are 0.

---

## 5. Speaker-Exclusive Dataset Splits

To prevent acoustic fingerprint leakage, all 30,999 clips are partitioned by speaker identity across the splits. There is **zero speaker overlap** between Train, Validation, and Test.

| Split | Clips | Percentage | Speakers | Multi-Label Events | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 17,555 | 56.6% | 18 | 658 | Model training & positive weight derivation |
| **Validation** | 6,632 | 21.4% | 126 | 260 | Model checkpoint selection & threshold tuning |
| **Test** | 6,812 | 22.0% | 137 | 254 | Final locked benchmark evaluation |
| **Total** | **30,999** | **100.0%** | **281** | **1,172** | Complete clean corpus |

### Pre-Flight Partition Integrity
- **High-Volume Hosts:** The 4 high-volume podcast hosts (Pamela Mertz, Daniele Rossi, Pedro Peña, Peter Reitzes) reside strictly in Train.
- **Guest Speakers:** Over 260 distinct podcast guests and interview participants are partitioned across Validation and Test.
- **TalkBank Participants:** Multiple sessions from the same TalkBank speaker (e.g., `24fa`, `24fb`) were clustered to unified IDs prior to splitting.
- **Panel Clips:** 156 multi-speaker panel clips from `StutteringIsCool` with indeterminate individual speaker attribution were quarantined exclusively into the Test set.

---

## 6. Manifest Files

The locked manifests are saved in `v2/dataset/manifests/`:
- `v2_full_clean_manifest.csv` (30,999 rows)
- `v2_train_manifest.csv` (17,555 rows)
- `v2_val_manifest.csv` (6,632 rows)
- `v2_test_manifest.csv` (6,812 rows)

Each manifest references audio via clean relative archive paths:
`clips/stuttering-clips/clips/<Show>_<EpId>_<ClipId>.wav`
