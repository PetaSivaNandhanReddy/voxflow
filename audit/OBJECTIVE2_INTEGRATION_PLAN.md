# VoxFlow V2 — Objective 2 Integration Plan

This document outlines how the session-level processing pipeline (Objective 2) from the original TARP project will be adapted and integrated into VoxFlow V2 after the core ML benchmark experiments (Models 1, 2, and 3) are complete.

---

## 1. What Objective 2 Does

Objective 2 expands VoxFlow from classifying isolated 3-second audio clips into analyzing **continuous speech sessions** (30–60+ seconds):

1. **Audio Ingestion & Validation:** Validates format (16 kHz mono PCM WAV), checks duration, and computes RMS energy to filter silent recordings.
2. **Sliding-Window Inference:** Slices continuous audio into 3.0-second overlapping windows with a 1.0-second step size.
3. **Multi-Label Window Classification:** Evaluates each window through the trained disfluency classifier (`Repetition`, `Prolongation`, `Block`).
4. **Temporal Event Aggregation:** Clusters contiguous overlapping windows that detect the same disfluency into discrete clinical events with explicit `[start_time, end_time]` timestamps, preventing double-counting artifacts caused by the 1-second sliding step.
5. **Session Fluency Summary:** Computes the **Session Fluency Ratio** (`fluent_windows / valid_windows`), total disfluent events by category, and an event timeline table.
6. **Persistence & Presentation:** Stores sessions, windows, and events in a local SQLite database and presents interactive clinical summaries via a REST API and Streamlit dashboard.

---

## 2. Key Files in TARP Containing Objective 2

| TARP Component | Path | Responsibility |
| :--- | :--- | :--- |
| **Session Engine** | `TARP/app/services/session_engine.py` | Validates audio, handles sliding-window slicing (3.0s window, 1.0s step) |
| **Event Aggregator** | `TARP/app/services/event_aggregator.py` | Merges contiguous overlapping windows into discrete events |
| **Session Summary** | `TARP/app/services/session_summary.py` | Computes summary metrics, fluency ratios, and event timelines |
| **Database Layer** | `TARP/app/db/database.py` | SQLite schema (`users`, `sessions`, `session_windows`, `session_events`, `session_summaries`) |
| **REST API** | `TARP/app/api/session_routes.py` | Flask routes (`POST /api/sessions/process`, `GET /api/sessions/{id}`) |
| **Dashboard** | `TARP/dashboard/dashboard.py` | Streamlit clinical monitoring interface |
| **Tests** | `TARP/tests/evaluation/run_final_objective2.py` | End-to-end continuous 42-second session validation script |

---

## 3. What Must Be Adapted for V2

The original TARP Objective 2 was built for a single-label 4-class softmax model (`Fluent`, `Repetition`, `Prolongation`, `Block`). VoxFlow V2 uses **multi-label classification with independent sigmoid thresholds**. The following components must be adapted:

1. **Downstream Fluent Definition:**
   - *TARP:* `Fluent` was an explicit softmax class.
   - *V2:* A window is defined as `Fluent` if and only if **all three** disfluency probabilities are below their respective locked validation thresholds:
     $$\text{Fluent} \iff (\hat{P}_{\text{rep}} < \tau_{\text{rep}}) \land (\hat{P}_{\text{pro}} < \tau_{\text{pro}}) \land (\hat{P}_{\text{blk}} < \tau_{\text{blk}})$$
2. **Multi-Label Event Aggregation:**
   - *TARP:* Clustered windows by a single winner-take-all `predicted_class`.
   - *V2:* Clusters events **independently** for each class stream (`Repetition`, `Prolongation`, `Block`). If a window contains both a repetition and a block, it contributes to both event timelines without conflict.
3. **Database Schema Update:**
   - In `session_windows`, store the raw sigmoid probabilities (`prob_repetition`, `prob_prolongation`, `prob_block`) and binary decisions against frozen thresholds.
4. **Decoupled Inference Engine:**
   - The inference engine must support plugging in any of the three V2 models:
     - Model 1: `MultiLabelSVMInference` (162 acoustic features)
     - Model 2: `HuBERTMultiLabelInference`
     - Model 3: `Wav2Vec2MultiLabelInference`

---

## 4. What Should NOT Be Migrated

- **Old V1 Model Weights:** Do not migrate `candidate_svm.joblib`, `candidate_2d-cnn.pt`, `candidate_cnn-gru.pt`, or `voxflow_model_v1.0.joblib`.
- **Old 646-Clip Data Manifests:** The old 646-clip dataset cohort is fully superseded by V2's 30,999 locked clean clips.
- **Legacy Batch Scripts:** Remove Windows-specific hardcoded `.bat` files (`run_voxflow_system.bat`).

---

## 5. Migration Sequence (Post-ML Benchmark)

```
[Phase 1: ML Models Complete]
  Model 1 (SVM) ✅ -> Model 2 (HuBERT) ⏳ -> Model 3 (Wav2Vec2) ⏳ -> Benchmark Selection
                                 ↓
[Phase 2: Core Engine Migration]
  Port and adapt session_engine.py & event_aggregator.py to v2/inference/
  Implement Multi-Label window classification with locked validation thresholds
                                 ↓
[Phase 3: Database & API]
  Set up v2/database/ with updated SQLite schema
  Port session_routes.py into v2/api/ using clean FastAPI or Flask
                                 ↓
[Phase 4: Clinical Dashboard]
  Port Streamlit dashboard into v2/dashboard/
  Display multi-label event timelines, session audio waveform, and fluency ratio trend
                                 ↓
[Phase 5: Validation & Transition]
  Execute end-to-end continuous session test on sample recordings
  Verify zero regressions, archive old TARP folder, and finalize repository
```

---

## 6. Compatibility & Testing Requirements

- **Continuous Audio Test:** Run a test session on continuous recordings (30–60s) to verify:
  1. Window count matches expected: $N_{\text{windows}} = \lfloor \frac{\text{duration} - 3.0}{1.0} \rfloor + 1$.
  2. Multi-label co-occurrences are properly clustered without double-counting.
  3. Fluency ratio is mathematically bounded in $[0.0, 1.0]$.
  4. SQLite records are fully populated with foreign key integrity.
