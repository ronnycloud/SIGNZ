# SignVision: Real-Time Sign Language Recognition & Translation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00C7B7.svg)](https://developers.google.com/mediapipe)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Computer Vision & Deep Learning system for real-time Sign Language Recognition (SLR) and continuous gesture translation using MediaPipe Holistic landmark extraction and Multi-Stream Temporal Attention Networks (MTAN).

---

## 🚀 Key Features

- **Multi-Stream Landmark Extraction**: Captures 454-dimensional feature vectors per frame combining 21 hand keypoints (both hands), 25 upper-body pose landmarks, inter-frame velocity vectors, and hand-presence indicators via MediaPipe Holistic.
- **Attention-Driven Deep Learning**:
  - **BiLSTM + Cross-Modal Attention**: Fuses spatial hand dynamics with body pose context.
  - **Temporal Self-Attention**: Captures long-range temporal dependencies across dynamic gesture sequences.
  - Also supports Transformer, CNN-LSTM, and baseline LSTM architectures via YAML configs.
- **Interactive Web Application**: Flask-based interface with live webcam feed, real-time bounding box/landmark overlay, continuous sentence translation, and latency/FPS telemetry via Server-Sent Events (SSE).
- **FastAPI Microservice**: High-throughput REST & WebSocket endpoints for remote client streaming and edge deployments.
- **ONNX Acceleration**: Export pipeline for 2–4× faster inference on CPUs and embedded devices.
- **Dataset Support**: Compatible with self-collected webcam gestures and the benchmark **WLASL** (Word-Level American Sign Language) dataset.

---

## 📁 Repository Structure

```text
Action_Detection_Project/
├── configs/                  # Modular YAML configuration files
│   ├── model_config.yaml     # Architecture hyperparameters (BiLSTM, Attention, dims)
│   ├── training_config.yaml  # Batch size, learning rates, epochs, scheduler
│   └── inference_config.yaml # Confidence thresholds, smoothing, buffer lengths
├── checkpoints/              # Trained model checkpoints (.pt, .onnx)
├── dataset/                  # Dataset directory (landmarks & splits)
├── scripts/                  # Optimization & utility scripts
│   ├── export_onnx.py        # PyTorch to ONNX export tool
│   └── benchmark.py          # Latency & throughput benchmarking
├── src/                      # Source code
│   ├── data/                 # PyTorch Datasets, data loaders & augmentations
│   ├── features/             # MediaPipe feature extractors, normalizers, velocity
│   ├── models/               # MTAN-SLR model architectures & attention modules
│   ├── nlp/                  # Gloss-to-English grammar translation
│   └── utils/                # Logging, visualization & helpers
├── static/ & templates/      # Web UI front-end assets
├── app.py                    # Flask full-stack web application
├── api_server.py             # FastAPI REST & WebSocket streaming server
├── train.py                  # PyTorch model training script
├── inference.py              # CLI / OpenCV real-time inference script
├── requirements.txt          # Python package dependencies
└── .gitignore                # Git ignore rules for datasets, models & venv
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Action_Detection_Project.git
cd Action_Detection_Project
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Quickstart & Usage

### 1. Run the Web Application
Launch the real-time recognition interface:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

### 2. Run Desktop Real-Time Inference
Run OpenCV webcam detection directly:
```bash
python inference.py
```

### 3. Run FastAPI Server
For REST API / WebSocket integration:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

---

## 🏋️ Model Training

### Option A: Custom Gesture Collection
1. Run data collection to record custom gestures via webcam:
   ```bash
   python 1_collect_data.py
   ```
2. Train the model:
   ```bash
   python train.py --epochs 50 --batch_size 16
   ```

### Option B: WLASL Dataset
Configure `configs/training_config.yaml` to point to your extracted WLASL keypoints and run:
```bash
python train.py --data_dir dataset/MP_Data --epochs 100
```

---

## ⚡ ONNX Export & Optimization

To export the trained PyTorch checkpoint to ONNX for accelerated inference:
```bash
python scripts/export_onnx.py --checkpoint checkpoints/best_model.pt --output checkpoints/model.onnx
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
The WLASL dataset is subject to its original academic terms and conditions.
