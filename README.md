# EdgeVision: Integrated Edge AI Suite

EdgeVision is a collection of high-performance computer vision pipelines optimized for heterogeneous hardware. This suite contains two primary engines: a multi-modal Face Recognition system and an Int8-quantized License Plate Recognition (LPR) system. Both are engineered to maintain High Definition video streams while performing asynchronous AI inference on edge-constrained devices like the Raspberry Pi 4.

---

## 1. Project Structure

```text
D:\EdgeVision\
├── FaceID/                # Biometric Recognition Engine
│   ├── .insightface/      # Auto-downloaded model binaries
│   ├── dataset/           # Raw identity photos
│   ├── faceeid.py         # Windows GPU runner
│   ├── try.py             # Multi-modal DB generator
│   └── rpi_faceid_*.py    # Multithreaded Pi streamer
├── LPR/                   # License Plate Recognition Engine
│   ├── inference.py       # Main NCNN + OCR application
│   ├── model_benchmark.py # Performance testing tool
│   └── pruned_int8.ncnn/  # YOLOv8 Quantized weights
├── README.md              # Global documentation
└── pyproject.toml         # Unified dependency management
```

---

## 2. Core Modules

### A. FaceID Engine
A biometric system utilizing the InsightFace framework for SCRFD detection and MobileFaceNet/ResNet recognition. 

*   **Architecture**: Decoupled threading (1080p Capture / 480p Inference).
*   **Database Configuration**: Uses DBSCAN clustering to generate multi-modal centroids, allowing one identity to have multiple mathematical "looks" (e.g., different lighting, accessories).
*   **Expansion Logic**: Employs a 1:5 aggressive augmentation strategy during database creation to ensure high accuracy with small localized datasets.
*   **Hardware Support**: Full CUDA acceleration for NVIDIA GTX 1050 Ti and optimized ONNX Runtime for Pi 4 CPU.

### B. LPR Engine (License Plate Recognition)
A real-time vehicle identification system using a custom YOLOv8 model pruned by 30% and quantized to Int8.

*   **Inference Backend**: Runs on the NCNN framework to utilize NEON vector instructions, reducing CPU overhead to approximately 55% on the Pi 4.
*   **Capture Pipeline**: Spawns a native subprocess (`rpicam-vid`) to read raw YUV420 bytes, bypassing OpenCV driver latency.
*   **OCR Integration**: Tesseract LSTM engine triggered asynchronously on high-resolution crops for maximum text extraction accuracy.
*   **Logging**: Debounced CSV logging logic to prevent redundant data entry.

---

## 3. Installation and Setup

This project uses `uv` for high-speed, isolated dependency management.

### System Dependencies (Linux/Pi 4)
```bash
sudo apt update
sudo apt install libgl1 tesseract-ocr libtesseract-dev -y
```

### Python Environment (Unified)
```powershell
# Create virtual environment
uv venv
.\.venv\Scripts\activate

# Install all required libraries for both engines
uv add insightface onnxruntime-gpu opencv-contrib-python flask scikit-learn ultralytics pytesseract ncnn numpy
```

---

## 4. Hardware Specifications

| Component | Minimum Requirement | Optimal Setup |
|:---|:---|:---|
| **Edge Processor** | Raspberry Pi 4 (2GB) | Raspberry Pi 4 (8GB) |
| **Windows GPU** | NVIDIA GTX 1050 | NVIDIA GTX 1050 Ti |
| **Camera 1** | USB Webcam (Logitech C920) | Raspberry Pi HQ Camera (IMX477) |
| **Cooling** | Passive Heatsinks | Active Cooling (Mandatory for NCNN) |

---

## 5. Usage Guide

### FaceID Workflow
1.  **Generate DB**: Place images in `FaceID/dataset/` and run `python try.py`.
2.  **Run Windows**: `python FaceID/faceeid.py`.
3.  **Run Edge**: `python FaceID/rpi_faceid_multithreaded.py`.

### LPR Workflow
1.  **Benchmarking**: Verify hardware performance using `python LPR/model_benchmark.py`.
2.  **Inference**: Start the system via `python LPR/inference.py`.

---

## 6. Networking and Streaming

Both engines serve an MJPEG stream via Flask. 
*   **Endpoint**: `http://<device_ip>:5000`
*   **Optimization**: Web streams are regulated to a 1024px width with 65% JPEG compression to maintain low-latency viewing over standard network bandwidth while the system continues internal high-resolution processing.

---

## 7. Git Management

This repository is optimized to prevent metadata corruption and bloat by ignoring large binary artifacts:
*   **Excluded**: `.venv/`, `.insightface/`, `models/`, `*.onnx`, `*.pkl`, `*.pth`, `*.csv`.
*   **Requirement**: On first deployment to a new device, re-run the database generation scripts to initialize local binary assets.