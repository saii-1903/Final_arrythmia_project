# Arrythmia AI Workstation: Active File Guide

These are the primary files that constitute the "Live" system. This list excludes temporary scripts, legacy backups, and purely experimental tests.

## 🌐 Web Dashboard & Frontend

| File Path | Language | Role |
| :--- | :--- | :--- |
| `dashboard/app.py` | Python (Flask) | The main web server. Handles routing, API requests, and database orchestration for the UI. |
| `dashboard/templates/index.html` | HTML / JS | The entire frontend UI. Contains the ECG plotting logic (D3.js) and the annotation interface. |

## 📡 Real-Time Streaming (gRPC)

| File Path | Language | Role |
| :--- | :--- | :--- |
| `grpc_server.py` | Python | The streaming server. Receives live ECG data from devices and returns real-time alerts. |
| `ecg.proto` | Protobuf | Defines the data structure for ECG samples and Arrhythmia alerts sent over the network. |

## 🧠 Decision Engine (The "Brain")

| File Path | Language | Role |
| :--- | :--- | :--- |
| `decision_engine/rhythm_orchestrator.py`| Python | The central coordinator that combines ML predictions with clinical rules to make a final diagnosis. |
| `decision_engine/rules.py` | Python | Contains the hard-coded clinical logic (e.g., "If PR > 200ms, then 1st Degree Block"). |
| `decision_engine/models.py` | Python | Defines the data structures (Event, Segment, DisplayState) used throughout the engine. |

## 🧪 Signal Processing & AI

| File Path | Language | Role |
| :--- | :--- | :--- |
| `signal_processing/cleaning.py` | Python | Removes noise, baseline wander, and powerline interference from raw signals. |
| `signal_processing/artifact_detection.py` | Python | Detects if a signal is too noisy or "artifacted" to be clinically reliable. |
| `xai/xai.py` | Python | Loads the Transformer model, performs inference, and generates saliency maps (XAI). |
| `database/db_service.py` | Python | The low-level database wrapper for all PostgreSQL operations. |

## 🏋️ Training & Configuration

| File Path | Language | Role |
| :--- | :--- | :--- |
| `models_training/data_loader.py` | Python | Handles the ingestion of massive ECG datasets for model training. |
| `models_training/retrain.py` | Python | The main entry point for retraining the Deep Learning model on new clinical data. |
| `requirements.txt` | Text | Lists all necessary Python libraries (NeuroKit2, PyTorch, Flask, etc.). |

---
*Note: The `scripts/` folder contains many test utilities (e.g., `test_pvc_logic.py`) which are important for development but are not part of the active production runtime.*
