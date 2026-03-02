# 🫀 ECG Arrhythmia AI Workstation

A state-of-the-art diagnostic platform for ECG arrhythmia detection, combining **Transformer-based Deep Learning** with **Clinical Rule Engines**.

## 🌟 Key Features

- **31+ Arrhythmia Classes**: Detection of everything from simple PACs/PVCs to complex patterns like Bigeminy, AFib, and Heart Blocks.
- **Real-Time Streaming**: High-performance gRPC server for low-latency live ECG analysis.
- **Explainable AI (XAI)**: Clinical justifications and saliency mapping for every diagnosis.
- **Expert-in-the-Loop**: Interactive dashboard for cardiologists to review, edit, and export clinical reports.
- **Sophisticated Orchestration**: Hybrid model that validates ML predictions against clinical rules (PR interval, HR, QRS width).

## 📂 Project Structure

- **`dashboard/`**: Flask-based Web UI for visualization and annotation.
- **`decision_engine/`**: The "Brain" - holds orchestration logic and clinical rules.
- **`grpc_server.py`**: Real-time bi-directional streaming interface.
- **`models_training/`**: Scripts for training and retraining the Transformer models.
- **`signal_processing/`**: Centralized cleaning, peak detection, and HRV feature extraction.
- **`xai/`**: Explainable AI components for clinical interpretation.

For a deeper dive into the technical architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- PostgreSQL (for data persistence)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Launch the System

**Start the Web Dashboard:**
```bash
cd dashboard
python app.py
```
Visit `http://localhost:5000`

**Start the gRPC Streaming Server:**
```bash
python grpc_server.py
```

## ⚕️ Supported Arrhythmias

| Category | Supported Types |
| :--- | :--- |
| **Baseline** | Sinus Rhythm, Bradycardia, Tachycardia, AFib, Flutter, Junctional |
| **Blocks** | 1st, 2nd (Mobitz I/II), and 3rd Degree AV Blocks |
| **Ectopy** | PACs, PVCs, Couplets, Runs |
| **Patterns** | Bigeminy, Trigeminy, Quadrigeminy |
| **Advanced** | VT, SVT, NSVT, PSVT |

## 🛠️ Development & Training

To retrain the model on new data:
```bash
cd models_training
python retrain.py --data_path ../data/your_dataset.json
```

---
*Developed for advanced cardiac signal analysis and AI-augmented clinical workflows.*
