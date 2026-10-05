# VoxFlow — Speech Fluency & Disfluency Monitoring System

VoxFlow is an end-to-end speech analysis system designed to detect and monitor stuttering disfluencies in continuous speech recordings. By pairing an accessible microcontroller-based microphone setup with a fine-tuned self-supervised speech representation model, VoxFlow identifies disfluency events, prevents duplicate counts across overlapping audio windows, and computes standardized fluency metrics for longitudinal tracking.

---

## 1. What VoxFlow Tackles

Traditional speech fluency evaluations rely heavily on manual counting of disfluencies during clinical sessions, which is time-intensive and subject to inter-rater variability. VoxFlow provides an automated, objective, and reproducible software workflow:

- **Multi-Label Disfluency Detection:** Detects three primary stuttering events—**Sound/Word Repetitions**, **Sound Prolongations**, and **Postural Inaudible Blocks**—allowing multiple disfluencies to co-occur in the same audio window.
- **Continuous Session Analysis:** Evaluates 30–60 second continuous speech sessions using overlapping sliding windows.
- **Temporal Event Clustering:** Groups contiguous and overlapping window detections into unified clinical events, ensuring a 3-second block spanning consecutive windows is counted as one event rather than multiple independent disfluencies.
- **Accessible Hardware Approach:** Designed to work with low-cost, off-the-shelf hardware (ESP32 Dev Module + INMP441 I2S MEMS microphone) streaming standard 16 kHz mono PCM audio to a host workstation.
- **Interactive Review:** Automatically persists sessions, window-level probabilities, and aggregated events into a local SQLite database, visualized through an interactive Streamlit dashboard.

---

## 2. System Architecture

VoxFlow connects hardware capture, deep learning inference, temporal post-processing, and interactive visualization into a unified pipeline:

```text
┌────────────────────────────────────────────────────────┐
│  Audio Ingestion (ESP32 Dev Module + INMP441 or WAV)  │
│  16 kHz Mono, 16-bit Linear PCM Audio Stream           │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Continuous Session Processing (30–60s Speech)         │
│  Sliding Window Slicing: 3.0s duration, 1.0s step      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  HuBERT Acoustic Representation Model                  │
│  Multi-Label Classification Head with Sigmoid Outputs  │
│  Calibrated Thresholds: Rep: 0.77 | Pro: 0.79 | Blk: 0.57 │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Temporal Event Aggregator                             │
│  Merge adjacent windows (gap <= 1.5s) per category     │
│  Computes Session Fluency Ratio = Fluent / Total Valid │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Local SQLite Persistence (voxflow.db)                 │
│  Relational storage: Sessions, Windows, Events         │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│  Flask REST API           │   │  Streamlit Dashboard      │
│  (port 5000)              │   │  (port 8501)              │
│  Full session lifecycle   │   │  Waveforms, Timelines,    │
│  start -> audio -> stop   │   │  History & Trend Tracking │
└───────────────────────────┘   └───────────────────────────┘
```

---

## 3. Achievements & Verified Results

All benchmarks were evaluated on a clean, unified dataset of **30,999 clips** (derived from SEP-28k, SEP-28k-Extended, and FluencyBank) under strict **speaker-exclusive partitioning** (zero speaker overlap between training, validation, and test cohorts).

### Verified Test Benchmark (6,812 Held-Out Unseen Speaker Clips)

| Metric | HuBERT (Production Model) | Wav2Vec2 | Baseline LinearSVC (MFCC) |
| :--- | :---: | :---: | :---: |
| **Macro F1** | **0.4599** | 0.4498 | 0.3121 |
| **Mean PR-AUC** | **0.4532** | 0.4288 | 0.2545 |
| **Mean ROC-AUC** | **0.8099** | 0.8020 | 0.6633 |
| **Repetition F1** | **0.5565** | 0.5322 | 0.3648 |
| **Prolongation F1** | **0.4841** | 0.4846 | 0.3604 |
| **Block F1** | **0.3393** | 0.3325 | 0.2533 |
| **Exact Subset Match** | **57.93%** | 55.09% | 30.11% |

### Key System Achievements
- **End-to-End Pipeline:** Complete 30–60 second session processing from raw audio input to database storage and dashboard visualization.
- **Robust Multi-Label Handling:** Independent Sigmoid outputs support real-world clinical co-occurrence of multiple disfluency types in the same phrase.
- **Non-Redundant Event Grouping:** Temporal aggregation prevents double-counting across overlapping 3-second windows.
- **Zero-ORM Relational Storage:** Automated SQLite database schema tracking sessions, individual window time series, and discrete clinical event boundaries.
- **Comprehensive Test Suite:** 17 automated software tests verifying inference, dataset integrity, aggregation logic, database operations, and REST endpoints.

---

## 4. Hardware Configuration & Wiring

VoxFlow is configured for an **ESP32 Dev Module** paired with an **INMP441 I2S digital omnidirectional MEMS microphone**.

### Pinout Connection Table

| INMP441 Pin | ESP32 Pin | Function |
| :--- | :--- | :--- |
| **VDD** | **3.3V** | Power (3.3V DC) |
| **GND** | **GND** | Ground |
| **SCK / BCK** | **GPIO 26** | I2S Bit Clock |
| **WS** | **GPIO 25** | I2S Word Select (Left/Right clock) |
| **SD / DATA** | **GPIO 32** | I2S Serial Data Out |
| **L/R** | **GND** | Channel Select (Ground = Left / Mono) |

### Hardware Audio & Communication Specifications
- **Sampling Rate:** 16,000 Hz
- **Channels:** 1 (Mono)
- **Data Format:** 16-bit Linear PCM
- **Serial Baud Rate:** 460,800 baud
- **Default Serial Port:** `COM8` (configurable via `VOXFLOW_SERIAL_PORT` environment variable or in `configs/config.py`)

> **Hardware Note:** Physical end-to-end validation on live microcontrollers is pending hardware availability. The complete software ingestion interface, byte streaming parser, and configuration bindings are implemented and verified.

---

## 5. Installation & Setup

### Requirements
- Python 3.10+
- macOS, Linux, or Windows
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/PetaSivaNandhanReddy/voxflow.git
cd voxflow
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Model Checkpoint Setup
VoxFlow requires the fine-tuned HuBERT checkpoint (`hubert_stage_B_best.pt`, ~493 MB). Due to GitHub's file size limit (100 MB), the model weights file is tracked in `.gitignore`.

Place the checkpoint in the designated model directory:
```text
models/Hubert_D/hubert_stage_B_best.pt
```

*(If you are setting up on a fresh machine without local weights, download `hubert_stage_B_best.pt` from the project's Kaggle / GitHub Releases repository and place it in `models/Hubert_D/`.)*

### 5. Configuration (Optional)
System settings are centralized in `configs/config.py`. You can adjust:
- Serial COM port or baud rate (or set `export VOXFLOW_SERIAL_PORT=/dev/ttyUSB0`)
- Analysis window length (default: 3.0s) and step size (default: 1.0s)
- Decision thresholds (locked: Repetition: 0.77, Prolongation: 0.79, Block: 0.57)

---

## 6. Running VoxFlow

### One-Command Startup (Recommended)
Launch both the Flask REST API backend and the Streamlit clinical dashboard with a single command:

```bash
python run_voxflow.py
```

- **REST API:** Running at `http://localhost:5000`
- **Dashboard:** Running at `http://localhost:8501` (opens automatically in your default browser)

Press `Ctrl+C` in your terminal to gracefully stop both services.

### Individual Service Commands (Optional)
If you prefer running services in separate terminals:

```bash
# Terminal 1: Launch REST API server
python server.py

# Terminal 2: Launch Streamlit dashboard
streamlit run dashboard.py
```

### Running Automated Tests
Run the complete test suite to verify model loading, database integrity, sliding-window generation, event aggregation, and REST endpoints:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 7. Using the Application

1. **Analyze Session Tab:** Upload a 30–60s WAV recording (or stream audio from the ESP32 setup) and click **Start Complete Session Analysis**.
2. **Current Session Tab:** Inspect the session timeline, view detected events (Repetitions, Prolongations, Blocks) plotted on an interactive Plotly chart, and audit individual 3.0s window probabilities.
3. **Session History Tab:** Browse historical recording sessions stored in the SQLite database, filter by user, and review progress.
4. **Fluency Trends Tab:** Track the **Session Fluency Ratio** ($0.0 \text{ to } 1.0$) across multiple sessions over time.
5. **Model Information Tab:** Review active model metadata, architecture details, and locked decision thresholds.

### REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/session/start` | Initialize a new speech monitoring session |
| `POST` | `/api/v1/session/<id>/audio` | Upload chunked raw PCM or multipart WAV audio |
| `POST` | `/api/v1/session/<id>/stop` | Conclude audio recording and flush buffer |
| `POST` | `/api/v1/session/<id>/analyze` | Run sliding-window inference, event aggregation, and DB save |
| `GET` | `/api/v1/session/<id>` | Retrieve full session summary, window probabilities, and events |
| `GET` | `/api/v1/session/latest` | Fetch the most recent analyzed session |
| `GET` | `/api/v1/model/info` | Inspect model architecture, class targets, and thresholds |
| `GET` | `/api/v1/health` | Health check endpoint |

---

## 8. Limitations & Scope

- **Research Prototype:** VoxFlow is an automated speech fluency research prototype designed for longitudinal tracking and progress monitoring. It is **not** a certified medical diagnostic device and does not make clinical diagnoses.
- **Physical Hardware Validation:** Physical testing on real ESP32 + INMP441 microcontrollers is pending hardware availability. The software pipeline and communication protocol are fully implemented.
- **Acoustic Environment Sensitivity:** Detection accuracy depends on microphone proximity, ambient background noise, and individual speaker idiosyncratic speech patterns.
- **Session-Based Scope:** The system analyzes standard continuous recording intervals (30–60 seconds) rather than claiming zero-latency instantaneous diagnosis.

---

## 9. Repository Structure

```text
voxflow/
├── README.md                 # Main project documentation
├── DATASET.md                # Dataset curation, filtering & split details
├── RESULTS.md                # Multi-model benchmark comparisons
├── requirements.txt          # Python dependencies
├── .gitignore                # Exclusion rules for checkpoints and large audio
│
├── run_voxflow.py            # One-command system launcher (API + Dashboard)
├── server.py                 # Flask REST API server entrypoint
├── dashboard.py              # Streamlit dashboard launcher entrypoint
│
├── configs/
│   ├── config.py             # Centralized system, hardware & model configuration
│   └── hubert_base_config.json # Offline HuBERT architecture configuration
│
├── inference/
│   └── engine.py             # Production HuBERT inference engine
│
├── app/
│   ├── api/
│   │   └── session_routes.py # Session lifecycle REST API blueprint
│   ├── db/
│   │   ├── database.py       # SQLite persistence layer (sessions, windows, events)
│   │   └── voxflow.db        # Auto-initialized local SQLite database
│   ├── services/
│   │   ├── session_engine.py # Sliding-window processing (3.0s window, 1.0s step)
│   │   ├── event_aggregator.py # Temporal disfluency event clustering
│   │   └── session_summary.py  # Session fluency metrics calculation
│   └── dashboard/
│       └── app.py            # Streamlit multi-page clinical interface
│
├── models/
│   ├── Hubert_D/             # Production HuBERT model (config, history, weights)
│   │   ├── config.json
│   │   ├── val_summary.json
│   │   └── hubert_stage_B_best.pt  # Model checkpoint (~493 MB, see Setup)
│   └── README.txt            # Model artifact inventory & index
│
├── tests/
│   ├── test_objective2.py    # End-to-end integration & REST API test suite
│   ├── test_inference.py     # Inference & temporal aggregation unit tests
│   ├── test_database.py      # SQLite database persistence tests
│   └── test_dataset.py       # Dataset split & manifest integrity tests
│
└── dataset/
    ├── manifests/            # Locked train, val, and test split manifests
    └── metadata/             # Official raw clip metadata CSVs
```
