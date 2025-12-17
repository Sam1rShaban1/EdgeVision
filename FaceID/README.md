# EdgeVision FaceID: Multi-Platform Biometric Recognition Engine

This repository contains a high-performance face recognition pipeline optimized for heterogeneous hardware environments. It supports hardware-accelerated inference on Windows (NVIDIA GPU) and resource-efficient deployment on Raspberry Pi 4 (CPU). The system utilizes the InsightFace framework for SCRFD detection and MobileFaceNet/ResNet recognition.

---

## 1. System Architecture

The engine is built on a decoupled, multithreaded architecture designed to resolve the bottleneck of processing high-resolution video on edge CPUs.

*   **Asynchronous Capture Thread**: Captures raw frames at 1920x1080 (HD) to maintain visual fidelity for the stream.
*   **Downscaled Inference Engine**: AI processing is conducted on a 640x480 sub-sampled frame to maintain target FPS on the Raspberry Pi 4.
*   **Coordinate Transformation**: Detection bounding boxes and facial landmarks are mathematically upscaled from the inference resolution back to the HD coordinate space before rendering.
*   **Multi-Modal Centroid Database**: Identity verification uses a clustered database model. Multiple "sub-centers" for each identity are generated using DBSCAN clustering, accounting for variations in lighting, facial hair, and accessories.

---

## 2. Technical Specifications

| Feature | Windows Implementation | Raspberry Pi 4 Implementation |
|:---|:---|:---|
| **Primary Model** | `buffalo_l` (ResNet50) | `buffalo_s` (MobileFaceNet) |
| **Acceleration** | CUDA (GTX 1050 Ti) | ONNX Runtime CPU |
| **Landmark Resolution** | 106-point mesh | 5-point alignment |
| **Inference FPS** | 60+ FPS | 12-18 FPS |
| **Stream Resolution** | 1920x1080 | 1280x720 / 1920x1080 |

---

## 3. Installation and Environment

This project uses `uv` for high-speed dependency management and reproducible virtual environments.

### Prerequisites
*   **Python**: 3.10 or 3.11 (refer to `.python-version`)
*   **Hardware**: NVIDIA GTX 1050 Ti (Windows) or Raspberry Pi 4 (8GB recommended)
*   **Drivers**: CUDA Toolkit 12.x and cuDNN 9.x (for Windows GPU support)

### Setup Commands
```powershell
# Create and activate virtual environment
uv venv
.\.venv\Scripts\activate

# Install core dependencies
uv add insightface onnxruntime-gpu opencv-contrib-python numpy flask scikit-learn
```

---

## 4. Operational Workflow

### Phase 1: Database Generation
Before running inference, you must generate a mathematical representation of your identities. Place raw images in a structured directory (`dataset/name/*.jpg`).

```powershell
# Edit try.py to select MODEL_NAME ("buffalo_l" or "buffalo_s")
python try.py
```
This script performs statistical outlier removal (standard deviation trimming) to ensure bad photos do not pollute the centroid.

### Phase 2: Live Inference (Windows)
Run the desktop-optimized runner for GPU-accelerated performance and GUI feedback.
```powershell
python faceeid.py
```
*   **Keybinds**:
    *   `D`: Toggle facial landmark visualization (Dots).
    *   `Q`: Terminate process.

### Phase 3: Edge Deployment (Pi 4)
Run the multithreaded Flask server to stream the recognition feed over the network.
```powershell
python rpi_faceid_multithreaded.py
```
Access the stream at `http://<rpi_ip>:5000`.

---

## 5. Directory Structure

```text
D:\EdgeVision\FaceID\
├── .insightface/       # AI model binaries (SCRFD, MobileFaceNet)
├── .venv/              # Isolated virtual environment
├── models/             # Local model storage
├── embeddings_*.pkl    # Serialized identity databases (L/S/SC)
├── faceeid.py          # Primary Windows GPU runner
├── rpi_faceid_*.py     # Multithreaded Flask streamer for Pi 4
├── try.py              # Advanced DB generator with DBSCAN clustering
└── pyproject.toml      # Project metadata and dependencies
```

---

## 6. Git Hygiene and Large File Management

To prevent repository corruption and bloat, this project ignores binary artifacts.
*   **Models**: `.insightface/` and `models/` are excluded.
*   **Databases**: All `*.pkl` files are excluded.
*   **Virtual Envs**: `.venv/` is excluded.

When deploying to a new machine, ensure you re-run the database generation script to recreate the local `.pkl` files based on your source imagery.