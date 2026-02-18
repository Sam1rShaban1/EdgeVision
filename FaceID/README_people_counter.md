# EdgeVision People Counting & Analytics System

An advanced foot traffic tracking and dwell time analysis system that builds on the EdgeVision FaceID engine to provide real-time people counting, movement tracking, and comprehensive analytics.

## 🎯 Features

### Core Functionality
- **Real-time People Counting**: Track individuals as they enter, move through, and exit monitored areas
- **Dwell Time Analysis**: Measure how long people spend in specific zones
- **Identity Recognition**: Leverage FaceID database to identify known individuals
- **Movement Tracking**: Visualize person paths and movement patterns
- **Zone-based Analytics**: Define counting zones for specific areas of interest

### Analytics Dashboard
- **Live Metrics**: Current occupancy, peak occupancy, average dwell time
- **Real-time Visualization**: Dual video feeds (raw + analytics overlay)
- **Interactive Charts**: Hourly traffic distribution and identity breakdown
- **Historical Data**: CSV logging for long-term analysis
- **Auto-refresh Dashboard**: Updates every 5 seconds

### Performance Optimized
- **Multi-threaded Architecture**: Separate threads for capture, inference, and logging
- **Raspberry Pi 4 Optimized**: Efficient resource usage for edge deployment
- **Configurable Performance**: Adjustable FPS, inference intervals, and tracking parameters

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (3.13 recommended)
- EdgeVision FaceID system with existing face database
- Raspberry Pi 4 B (8GB recommended) or equivalent hardware
- Camera module (Pi HQ Camera or USB webcam)

### Installation

1. **Install Dependencies**
```bash
cd EdgeVision/FaceID
pip install -r requirements_people_counter.txt
```

2. **Verify FaceID Database**
```bash
# Ensure you have a face database generated
python try.py  # This creates embeddings_buffalo_sc.pkl
```

3. **Run Tests**
```bash
python test_people_counter.py
```

4. **Start the System**
```bash
python people_counter.py
```

5. **Access Dashboard**
Open your browser and navigate to: `http://<your-pi-ip>:5001`

## 📊 Dashboard Overview

### Metrics Grid
- **Current Occupancy**: Number of people currently in frame
- **Total People Today**: Cumulative count for the day
- **Peak Occupancy**: Maximum simultaneous occupancy recorded
- **Average Dwell Time**: Mean time people spend in the monitoring area
- **In Counting Zone**: People currently in the designated counting zone
- **Active Tracks**: Number of people being actively tracked

### Video Feeds
- **Live Video Feed**: Raw camera input with real-time processing
- **Analytics View**: Video overlay with tracking trails, zones, and metrics

### Charts
- **Hourly Traffic Distribution**: Bar chart showing traffic patterns throughout the day
- **Identity Distribution**: Doughnut chart of known vs unknown individuals

## ⚙️ Configuration

### Key Settings (in `people_counter.py`)

```python
CONFIG = {
    # Camera Settings
    "STREAM_WIDTH": 1920,
    "STREAM_HEIGHT": 1080,
    "FRAMERATE": 25,
    
    # Performance Settings
    "INFER_FPS": 4,  # Face detection frequency
    "STREAM_FPS": 10,  # Web stream FPS
    
    # People Counting Settings
    "DWELL_TIME_THRESHOLD": 30,  # seconds to count as "dwelling"
    "COUNTING_ZONE": {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.8},
    "MAX_TRACKING_AGE": 60,  # seconds to keep inactive tracks
    
    # Web Interface
    "FLASK_PORT": 5001,
}
```

### Counting Zone Configuration
The counting zone is defined in normalized coordinates (0.0 to 1.0):
- `x`, `y`: Top-left corner position
- `width`, `height`: Zone dimensions

Example: Center zone covering 60% of frame width and 80% of height:
```python
"COUNTING_ZONE": {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.8}
```

## 🔧 Architecture

### Multi-threaded Design
1. **Video Capture Thread**: High-performance camera input using `rpicam-vid`
2. **Inference Thread**: Face detection and recognition at configurable intervals
3. **Logging Thread**: Asynchronous CSV logging of all events
4. **Web Server Thread**: Flask application serving dashboard and video streams

### Data Flow
```
Camera → Capture Thread → Frame Buffer
                              ↓
Inference Thread → Face Detection → Track Updates → Analytics
                              ↓
Logging Thread ← Track Updates ← CSV Events
                              ↓
Web Server ← Analytics ← Video Streams
```

### Tracking Algorithm
- **Detection**: InsightFace SCRFD for face detection
- **Recognition**: MobileFaceNet embeddings with database matching
- **Tracking**: Position-based track association with distance thresholds
- **Zone Detection**: Real-time position checking against counting zones

## 📈 Analytics & Data

### Real-time Metrics
- Occupancy counting with entry/exit detection
- Dwell time calculation per person
- Zone-based presence tracking
- Identity distribution analysis

### Historical Data
All events are logged to `people_analytics.csv` with columns:
- `timestamp`: Event time
- `event`: Type (person_entered, person_exited)
- `track_id`: Unique tracking identifier
- `identity`: Recognized person name (if available)
- `dwell_time`: Time spent in monitoring area

### Performance Monitoring
- CPU usage optimization for edge devices
- Memory-efficient track management
- Configurable inference intervals
- Thermal-aware processing

## 🎨 Customization

### Adding New Analytics
Extend the `AnalyticsData` class to include custom metrics:

```python
@dataclass
class AnalyticsData:
    # ... existing metrics ...
    custom_metric: float = 0.0
    
    def update_custom_metric(self):
        # Your custom logic here
        pass
```

### Custom Zones
Define multiple counting zones for complex areas:

```python
MULTI_ZONES = {
    "entrance": {"x": 0.0, "y": 0.4, "width": 0.2, "height": 0.2},
    "main_area": {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.8},
    "exit": {"x": 0.8, "y": 0.4, "width": 0.2, "height": 0.2},
}
```

### Dashboard Customization
Modify the `DASHBOARD_HTML` template to add:
- New chart types
- Additional metric displays
- Custom styling
- Interactive controls

## 🐛 Troubleshooting

### Common Issues

**Low FPS on Pi 4**
- Reduce `INFER_FPS` in config
- Lower camera resolution
- Ensure proper cooling

**Tracking Loss**
- Increase tracking distance threshold
- Adjust `MAX_TRACKING_AGE`
- Verify lighting conditions

**High CPU Usage**
- Reduce inference frequency
- Lower stream resolution
- Monitor thermal throttling

**Dashboard Not Loading**
- Check Flask port configuration
- Verify network connectivity
- Check browser console for errors

### Performance Tuning

```bash
# Monitor system resources
watch -n 2 'vcgencmd measure_temp && ps aux | grep python'

# Check GPU memory (if available)
nvidia-smi

# Monitor disk space for logs
df -h
```

## 🔒 Security & Privacy

### Data Protection
- All processing happens locally on the edge device
- No facial data is transmitted externally
- CSV logs contain timestamps and track IDs only

### Access Control
- Dashboard runs on local network by default
- Configure firewall rules as needed
- Consider adding authentication for production use

## 🚀 Future Enhancements

### Planned Features
- **Multi-camera Support**: Track across multiple camera feeds
- **3D Tracking**: Depth camera integration for spatial analytics
- **Behavior Analysis**: Anomaly detection and pattern recognition
- **Mobile App**: Remote monitoring and alerts
- **Cloud Integration**: Optional cloud analytics and backup

### API Development
- RESTful API for integration with other systems
- WebSocket support for real-time data streaming
- Webhook system for event notifications

## 📝 License

This system is part of the EdgeVision suite. See main project license for details.

## 🤝 Contributing

Contributions welcome! Please follow the main project's contribution guidelines and test thoroughly before submitting pull requests.

---

**System Requirements**: Minimum Raspberry Pi 4 B (4GB), recommended 8GB for optimal performance with multiple concurrent tracks.