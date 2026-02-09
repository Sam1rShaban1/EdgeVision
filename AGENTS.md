# EdgeVision: Agent Development Guide

This guide provides essential information for AI agents working on the EdgeVision project - a high-performance computer vision suite optimized for Raspberry Pi 4 B (8GB) and heterogeneous hardware platforms.

## Project Overview

EdgeVision consists of two primary engines:
- **FaceID**: Multi-modal biometric recognition using InsightFace framework
- **PiLPR**: Real-time License Plate Recognition using quantized YOLOv8 + NCNN

Both systems are engineered for HD video streaming with asynchronous AI inference on edge-constrained devices.

*Note: The main README refers to this as "LPR" but the actual directory is named "PiLPR". This guide uses the actual directory name "PiLPR".*

---

## 1. Build and Development Commands

### Environment Setup
```bash
# Create and activate virtual environment (recommended: uv)
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies for FaceID
cd FaceID && uv sync

# Install dependencies for PiLPR (manual for now)
cd PiLPR && pip install ultralytics opencv-python-headless pytesseract flask psutil numpy ncnn
```

### Code Quality and Formatting
```bash
# Run Ruff formatter and linter (project standard)
ruff check .                    # Check for issues
ruff check . --fix             # Auto-fix issues
ruff format .                  # Format code

# Run both formatting and linting
ruff check . --fix && ruff format .
```

### Testing Commands

#### FaceID Testing
```bash
# Test database generation
cd FaceID
python try.py                  # Interactive DB generation with clustering

# Benchmark face recognition performance
python -c "
import time, cv2
from faceeid import FaceIDRunner
runner = FaceIDRunner()
start = time.time()
# Run 100 face detections for benchmark
for i in range(100):
    frame = cv2.imread('test_image.jpg')
    results = runner.process_frame(frame)
print(f'Avg inference time: {(time.time()-start)/100:.3f}s')
"

# Test individual components
python -c "
import pickle, numpy as np
from sklearn.metrics import pairwise_distances
# Test embedding similarity
with open('embeddings_buffalo_s.pkl', 'rb') as f:
    db = pickle.load(f)
print(f'Database contains {len(db)} identities')
"
```

#### PiLPR Testing
```bash
# Model performance benchmarking
cd PiLPR
python model_benchmark.py      # Comprehensive performance testing

# Test individual inference backends
python -c "
import time, cv2
from ultralytics import YOLO
model = YOLO('pruned_ncnn_model/model.ncnn')
start = time.time()
results = model('test_image.jpg', imgsz=416)
print(f'Inference time: {(time.time()-start)*1000:.1f}ms')
"

# Test OCR pipeline
python -c "
import cv2, pytesseract
img = cv2.imread('plate_crop.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray, config='--psm 7')
print(f'OCR result: {text.strip()}')
"
```

#### Integration Testing
```bash
# Test full FaceID pipeline (Windows)
cd FaceID
python faceeid.py             # GUI-based testing with D key toggle

# Test full FaceID pipeline (Pi 4)
python rpi_faceid_multithreaded.py &
curl http://localhost:5000   # Test MJPEG stream

# Test full PiLPR pipeline
cd PiLPR
python inference.py &
curl http://localhost:5000   # Test MJPEG stream
```

### Performance Profiling
```bash
# CPU/Memory profiling
python -m cProfile -o profile.stats faceeid.py
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# Thermal monitoring (Pi 4)
watch -n 2 'vcgencmd measure_temp && ps aux | grep python'
```

---

## 2. Code Style Guidelines

### Import Organization
Follow this strict order in all Python files:

```python
# 1. Standard library imports
import os
import time
import threading
import queue
from typing import Dict, List, Optional

# 2. Third-party imports
import cv2
import numpy as np
import pickle
from flask import Flask, Response
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

# 3. Local imports
from config import CONFIG
from utils import preprocess_image
```

### Naming Conventions
- **Variables**: `snake_case` (e.g., `shared_data`, `last_infer_time`)
- **Functions**: `snake_case` (e.g., `generate_frames()`, `process_detections()`)
- **Classes**: `PascalCase` (e.g., `FaceIDRunner`, `VideoCaptureThread`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `CONFIG`, `MODEL_PATH`, `MAX_THREADS`)

### Code Formatting Standards
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Maximum 100 characters
- **Spacing**: Single space around operators, no space after opening parentheses
- **Blank lines**: Use blank lines to separate logical sections

### Configuration Management
```python
# Use centralized dictionary-based configuration
CONFIG = {
    "DB_PATH": f"embeddings_{CURRENT_MODEL}.pkl",
    "THRESHOLD": 0.50,
    "CAMERA_ID": 0,
    "FRAME_WIDTH": 1080,
    "FRAME_HEIGHT": 720,
    "INFERENCE_INTERVAL": 0.1,  # seconds
    "MAX_THREADS": 2,  # Pi 4 optimization
}
```

---

## 3. Error Handling Patterns

### Defensive Programming Approach
```python
# Always validate before operations
if not os.path.exists(CONFIG["DB_PATH"]):
    print(f"Error: Database file '{CONFIG['DB_PATH']}' not found.")
    exit(1)

# Use graceful fallbacks for hardware dependencies
try:
    self.app.prepare(ctx_id=0, det_size=(640, 640))
    print("Inference Engine: GPU (CUDA) Initialized.")
except Exception as e:
    print(f"Warning: GPU Init Failed ({e}). Falling back to CPU.")
    self.app.prepare(ctx_id=-1, det_size=(640, 640))
```

### Threading Error Handling
```python
# Always use locks for shared state
class SharedState:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = True
    
    def update_frame(self, new_frame):
        with self.frame_lock:
            self.frame = new_frame.copy()
```

---

## 4. Performance Optimization Guidelines

### Raspberry Pi 4 Specific Optimizations
```python
# Limit CPU threads for Pi 4
so = ort.SessionOptions()
so.intra_op_num_threads = 2
so.inter_op_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Use appropriate model sizes
PI_MODEL = "buffalo_s"  # MobileFaceNet (faster)
DESKTOP_MODEL = "buffalo_l"  # ResNet50 (more accurate)
```

### Memory Management
```python
# Always copy frames when crossing thread boundaries
def safe_frame_update(self, frame):
    with self.lock:
        self.shared_frame = frame.copy()  # Prevent memory corruption

# Use queues for thread-safe communication
self.frame_queue = queue.Queue(maxsize=2)  # Limit memory usage
```

### Rate Limiting for Real-time Performance
```python
# Control inference intervals
last_infer_time = 0
INFER_INTERVAL = 0.1  # 10 FPS max

current_time = time.time()
if current_time - last_infer_time > INFER_INTERVAL:
    # Run inference
    last_infer_time = current_time
```

---

## 5. Testing and Validation

### Model Validation
```python
# Always validate model files before use
def validate_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Test model loading
    try:
        model = YOLO(model_path)
        return True
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False
```

### Performance Benchmarking
```python
# Use the established benchmarking pattern
def benchmark_inference(model, test_image, iterations=100):
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = model(test_image)
        times.append((time.time() - start) * 1000)
    
    return {
        'avg_ms': np.mean(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'std_ms': np.std(times),
        'fps': 1000 / np.mean(times)
    }
```

### Integration Testing
```python
# Test full pipeline with mock data
def test_face_recognition_pipeline():
    # Load test database
    with open('embeddings_buffalo_s.pkl', 'rb') as f:
        db = pickle.load(f)
    
    # Test with known face
    test_embedding = np.random.rand(512)  # Mock embedding
    # Verify pipeline components work end-to-end
```

---

## 6. Hardware-Specific Guidelines

### Raspberry Pi 4 Optimizations
- **CPU**: Limit to 2-3 threads for AI inference
- **Memory**: Use 8GB model, monitor with `free -h`
- **Thermal**: Active cooling required above 75°C
- **Camera**: Use `rpicam-vid` subprocess for low latency

### Windows GPU Optimizations
- **CUDA**: Prefer GTX 1050 Ti or better
- **Memory**: Monitor GPU memory usage
- **Models**: Use larger models (`buffalo_l`) when GPU available

### Cross-Platform Compatibility
```python
# Detect hardware and adapt automatically
import platform

def get_optimal_config():
    if platform.system() == "Linux" and "arm" in platform.machine():
        return PI_CONFIG
    else:
        return DESKTOP_CONFIG
```

*Note: This pattern is suggested for future enhancements. Current codebase uses manual configuration.*

---

## 7. Security and Production Guidelines

### Input Validation
```python
# Always validate camera inputs
def validate_frame(frame):
    if frame is None:
        return False
    if frame.size == 0:
        return False
    if len(frame.shape) != 3:
        return False
    return True
```

### Resource Cleanup
```python
# Always cleanup resources in finally blocks
try:
    cap = cv2.VideoCapture(CONFIG["CAMERA_ID"])
    # ... processing ...
finally:
    cap.release()
    cv2.destroyAllWindows()
```

### Logging Best Practices
```python
# Use structured logging for production
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edgevision.log'),
        logging.StreamHandler()
    ]
)
```

---

## 8. Development Workflow

### Before Making Changes
1. Run `ruff check . --fix && ruff format .`
2. Test with `python model_benchmark.py`
3. Verify thermal performance on Pi 4

### After Making Changes
1. Run full test suite
2. Benchmark performance impact
3. Test on both Pi 4 and Windows if applicable
4. Update documentation if needed

### Git Hygiene
- Binary files are excluded (.pkl, .onnx, .pth, .csv)
- Models must be regenerated on new deployments
- Use descriptive commit messages with component prefixes

---

## 9. Troubleshooting Common Issues

### Pi 4 Specific
- **Thermal throttling**: Monitor with `vcgencmd measure_temp`
- **Camera issues**: Use `rpicam-vid` instead of cv2.VideoCapture
- **Memory**: Use `free -h` to monitor RAM usage

### Model Loading
- **InsightFace**: Models auto-download to `.insightface/`
- **NCNN**: Ensure both `.param` and `.bin` files present
- **ONNX**: Check GPU/CPU runtime compatibility

### Performance Issues
- **FPS drops**: Check thread competition and CPU usage
- **Memory leaks**: Monitor with `ps aux --sort=%mem`
- **Network latency**: Optimize MJPEG compression settings

---

This guide serves as the definitive reference for agents working on EdgeVision. Follow these conventions to maintain code quality, performance, and reliability across all development efforts.