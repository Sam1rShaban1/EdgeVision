#!/usr/bin/env python3
"""
EdgeVision People Counting & Analytics System
Builds on FaceID to add foot traffic tracking, dwell time analysis, and real-time dashboard
"""

import cv2
import numpy as np
import pickle
import os
import time
import threading
import queue
import json
import csv
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from flask import Flask, Response, render_template_string, jsonify
from insightface.app import FaceAnalysis
import subprocess
from picamera2 import Picamera2
import onnxruntime as ort

# =========================================================================
# CONFIGURATION
# =========================================================================
CURRENT_MODEL = "buffalo_sc"
CONFIG = {
    "DB_PATH": f"embeddings_{CURRENT_MODEL}.pkl",
    "THRESHOLD": 0.50,
    "STREAM_WIDTH": 1920,
    "STREAM_HEIGHT": 1080,
    "INFER_WIDTH": 640,
    "INFER_HEIGHT": 480,
    "PEOPLE_LOG_PATH": "people_analytics.csv",
    "FLASK_PORT": 5001,  # Different port to avoid conflict
    "FRAMERATE": 25,
    "INFER_FPS": 4,
    "STREAM_FPS": 10,
    # People counting specific
    "DWELL_TIME_THRESHOLD": 30,  # seconds to count as "dwelling"
    "COUNTING_ZONE": {
        "x": 0.2,
        "y": 0.1,
        "width": 0.6,
        "height": 0.8,
    },  # normalized coords
    "MAX_TRACKING_AGE": 60,  # seconds to keep tracking inactive person
}


# =========================================================================
# DATA STRUCTURES
# =========================================================================
@dataclass
class PersonTrack:
    """Track a person across frames"""

    track_id: int
    identity: Optional[str] = None
    first_seen: datetime = None
    last_seen: datetime = None
    positions: deque = None  # Recent positions for movement analysis
    dwell_time: float = 0.0
    in_counting_zone: bool = False
    crossing_count: int = 0  # Number of times crossed counting line

    def __post_init__(self):
        if self.positions is None:
            self.positions = deque(maxlen=50)
        if self.first_seen is None:
            self.first_seen = datetime.now()
        if self.last_seen is None:
            self.last_seen = datetime.now()


@dataclass
class AnalyticsData:
    """Real-time analytics data"""

    total_people_today: int = 0
    current_occupancy: int = 0
    peak_occupancy: int = 0
    average_dwell_time: float = 0.0
    people_in_zone: int = 0
    crossing_count: int = 0
    hourly_counts: Dict[int, int] = None
    identity_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.hourly_counts is None:
            self.hourly_counts = defaultdict(int)
        if self.identity_distribution is None:
            self.identity_distribution = defaultdict(int)


# =========================================================================
# SHARED STATE
# =========================================================================
class SharedState:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()
        self.tracks = {}  # track_id -> PersonTrack
        self.tracks_lock = threading.Lock()
        self.analytics = AnalyticsData()
        self.analytics_lock = threading.Lock()
        self.next_track_id = 1
        self.log_queue = queue.Queue()
        self.running = True


STATE = SharedState()


# =========================================================================
# PEOPLE COUNTING ENGINE
# =========================================================================
class PeopleCounter:
    """Core people counting and tracking logic"""

    def __init__(self, face_analyzer):
        self.face_analyzer = face_analyzer
        self.last_positions = {}  # track_id -> (x, y, timestamp)

    def update_tracks(self, detections, frame_shape):
        """Update person tracks with new detections"""
        current_time = datetime.now()
        new_tracks = {}

        # Match detections to existing tracks
        for det in detections:
            bbox = det["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Find best matching existing track
            best_match_id = None
            best_distance = float("inf")

            with STATE.tracks_lock:
                for track_id, track in STATE.tracks.items():
                    if track.positions:
                        last_pos = track.positions[-1]
                        distance = np.sqrt(
                            (center_x - last_pos[0]) ** 2
                            + (center_y - last_pos[1]) ** 2
                        )
                        if (
                            distance < best_distance and distance < 100
                        ):  # 100 pixel threshold
                            best_distance = distance
                            best_match_id = track_id

            if best_match_id is not None:
                # Update existing track
                track = STATE.tracks[best_match_id]
                track.positions.append((center_x, center_y, current_time))
                track.last_seen = current_time
                track.dwell_time = (current_time - track.first_seen).total_seconds()

                # Update identity if available
                if det.get("identity") and det["identity"] != "Unknown":
                    track.identity = det["identity"]

                new_tracks[best_match_id] = track
            else:
                # Create new track
                track = PersonTrack(
                    track_id=STATE.next_track_id,
                    identity=det.get("identity"),
                    first_seen=current_time,
                    last_seen=current_time,
                )
                track.positions.append((center_x, center_y, current_time))
                new_tracks[STATE.next_track_id] = STATE.next_track_id
                STATE.next_track_id += 1

                # Log new person
                STATE.log_queue.put(
                    {
                        "event": "person_entered",
                        "track_id": track.track_id,
                        "identity": track.identity,
                        "timestamp": current_time.isoformat(),
                    }
                )

        # Remove old tracks
        with STATE.tracks_lock:
            for track_id in list(STATE.tracks.keys()):
                if track_id not in new_tracks:
                    track = STATE.tracks[track_id]
                    age = (current_time - track.last_seen).total_seconds()
                    if age > CONFIG["MAX_TRACKING_AGE"]:
                        # Log person exit
                        STATE.log_queue.put(
                            {
                                "event": "person_exited",
                                "track_id": track.track_id,
                                "identity": track.identity,
                                "dwell_time": track.dwell_time,
                                "timestamp": current_time.isoformat(),
                            }
                        )
                        del STATE.tracks[track_id]
                    else:
                        new_tracks[track_id] = track

            STATE.tracks.update(new_tracks)

        # Update analytics
        self._update_analytics()

    def _update_analytics(self):
        """Update real-time analytics"""
        current_time = datetime.now()

        with STATE.analytics_lock:
            # Current occupancy
            STATE.analytics.current_occupancy = len(STATE.tracks)
            STATE.analytics.peak_occupancy = max(
                STATE.analytics.peak_occupancy, STATE.analytics.current_occupancy
            )

            # People in counting zone
            zone_people = 0
            for track in STATE.tracks.values():
                if track.positions and self._is_in_counting_zone(track.positions[-1]):
                    zone_people += 1
            STATE.analytics.people_in_zone = zone_people

            # Average dwell time
            if STATE.tracks:
                avg_dwell = np.mean(
                    [track.dwell_time for track in STATE.tracks.values()]
                )
                STATE.analytics.average_dwell_time = avg_dwell

            # Hourly counts
            hour = current_time.hour
            STATE.analytics.hourly_counts[hour] = STATE.analytics.current_occupancy

            # Identity distribution
            STATE.analytics.identity_distribution.clear()
            for track in STATE.tracks.values():
                if track.identity:
                    STATE.analytics.identity_distribution[track.identity] += 1

    def _is_in_counting_zone(self, position):
        """Check if position is in counting zone"""
        x, y, _ = position
        zone = CONFIG["COUNTING_ZONE"]
        frame_w, frame_h = CONFIG["STREAM_WIDTH"], CONFIG["STREAM_HEIGHT"]

        zone_x = zone["x"] * frame_w
        zone_y = zone["y"] * frame_h
        zone_w = zone["width"] * frame_w
        zone_h = zone["height"] * frame_h

        return zone_x <= x <= zone_x + zone_w and zone_y <= y <= zone_y + zone_h


# =========================================================================
# VIDEO CAPTURE THREAD
# =========================================================================
class VideoCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        try:
            # Use rpicam-vid for better performance
            cmd = [
                "rpicam-vid",
                "-t",
                "0",
                "--inline",
                "--width",
                str(CONFIG["STREAM_WIDTH"]),
                "--height",
                str(CONFIG["STREAM_HEIGHT"]),
                "--framerate",
                str(CONFIG["FRAMERATE"]),
                "--codec",
                "mjpeg",
                "-o",
                "-",
            ]

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

            while self.running and STATE.running:
                raw_frame = process.stdout.read(
                    CONFIG["STREAM_WIDTH"] * CONFIG["STREAM_HEIGHT"] * 3
                )
                if (
                    len(raw_frame)
                    != CONFIG["STREAM_WIDTH"] * CONFIG["STREAM_HEIGHT"] * 3
                ):
                    continue

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (CONFIG["STREAM_HEIGHT"], CONFIG["STREAM_WIDTH"], 3)
                )

                with STATE.frame_lock:
                    STATE.frame = frame.copy()

        except Exception as e:
            print(f"Camera error: {e}")
            # Fallback to OpenCV
            self._fallback_capture()

    def _fallback_capture(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["STREAM_WIDTH"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["STREAM_HEIGHT"])

        while self.running and STATE.running:
            ret, frame = cap.read()
            if ret:
                with STATE.frame_lock:
                    STATE.frame = frame.copy()

        cap.release()


# =========================================================================
# INFERENCE THREAD
# =========================================================================
class InferenceThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.people_counter = None

    def run(self):
        # Initialize face analyzer
        print("Initializing People Counter...")
        app = FaceAnalysis(name=CONFIG["DETECTOR_MODEL"], root=".")
        app.prepare(ctx_id=-1, det_size=(CONFIG["INFER_WIDTH"], CONFIG["INFER_HEIGHT"]))

        # Load database
        if os.path.exists(CONFIG["DB_PATH"]):
            with open(CONFIG["DB_PATH"], "rb") as f:
                self.db = pickle.load(f)
            print(f"Loaded {len(self.db)} identities")
        else:
            self.db = {}
            print("No identity database found")

        self.people_counter = PeopleCounter(app)

        last_infer = 0
        infer_interval = 1.0 / CONFIG["INFER_FPS"]

        while STATE.running:
            current_time = time.time()

            # Rate limiting
            if current_time - last_infer < infer_interval:
                time.sleep(0.01)
                continue

            with STATE.frame_lock:
                if STATE.frame is None:
                    time.sleep(0.1)
                    continue
                frame = STATE.frame.copy()

            # Run inference
            infer_frame = cv2.resize(
                frame, (CONFIG["INFER_WIDTH"], CONFIG["INFER_HEIGHT"])
            )
            faces = app.get(infer_frame)

            # Process detections
            detections = []
            for face in faces:
                bbox = face["bbox"]
                # Scale coordinates back to original frame
                scale_x = CONFIG["STREAM_WIDTH"] / CONFIG["INFER_WIDTH"]
                scale_y = CONFIG["STREAM_HEIGHT"] / CONFIG["INFER_HEIGHT"]

                scaled_bbox = [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y,
                ]

                # Identity recognition
                identity = "Unknown"
                if face["embedding"] is not None and len(self.db) > 0:
                    min_dist = float("inf")
                    for name, embeddings in self.db.items():
                        for emb in embeddings:
                            dist = np.linalg.norm(face["embedding"] - emb)
                            if dist < min_dist and dist < CONFIG["THRESHOLD"]:
                                min_dist = dist
                                identity = name

                detections.append(
                    {
                        "bbox": scaled_bbox,
                        "identity": identity,
                        "confidence": face.get("det_score", 0.0),
                    }
                )

            # Update tracks
            self.people_counter.update_tracks(detections, frame.shape)

            # Store detections for rendering
            with STATE.tracks_lock:
                STATE.latest_detections = detections

            last_infer = current_time


# =========================================================================
# LOGGING THREAD
# =========================================================================
class LoggingThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        csv_file = CONFIG["PEOPLE_LOG_PATH"]

        # Initialize CSV with headers
        file_exists = os.path.exists(csv_file)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["timestamp", "event", "track_id", "identity", "dwell_time"]
                )

        while STATE.running:
            try:
                log_entry = STATE.log_queue.get(timeout=1.0)
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            log_entry["timestamp"],
                            log_entry["event"],
                            log_entry.get("track_id", ""),
                            log_entry.get("identity", ""),
                            log_entry.get("dwell_time", ""),
                        ]
                    )
            except queue.Empty:
                continue


# =========================================================================
# WEB INTERFACE
# =========================================================================
app = Flask(__name__)

# HTML Template for Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>EdgeVision People Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            background: #1a1a1a; 
            color: #fff; 
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #333;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-label {
            font-size: 0.9em;
            color: #ccc;
            margin-top: 5px;
        }
        .video-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .video-box {
            background: #2a2a2a;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        .video-box img {
            max-width: 100%;
            border-radius: 5px;
        }
        .chart-container {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .refresh-info {
            text-align: center;
            color: #888;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 EdgeVision People Analytics</h1>
        <p>Real-time foot traffic tracking and dwell time analysis</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value" id="current-occupancy">0</div>
            <div class="metric-label">Current Occupancy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="total-today">0</div>
            <div class="metric-label">Total People Today</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="peak-occupancy">0</div>
            <div class="metric-label">Peak Occupancy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="avg-dwell">0s</div>
            <div class="metric-label">Avg Dwell Time</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="people-in-zone">0</div>
            <div class="metric-label">In Counting Zone</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="active-tracks">0</div>
            <div class="metric-label">Active Tracks</div>
        </div>
    </div>

    <div class="video-container">
        <div class="video-box">
            <h3>Live Video Feed</h3>
            <img src="/video_feed" alt="Live feed">
        </div>
        <div class="video-box">
            <h3>Analytics View</h3>
            <img src="/analytics_feed" alt="Analytics overlay">
        </div>
    </div>

    <div class="chart-container">
        <h3>Hourly Traffic Distribution</h3>
        <canvas id="hourlyChart" width="400" height="200"></canvas>
    </div>

    <div class="chart-container">
        <h3>Identity Distribution</h3>
        <canvas id="identityChart" width="400" height="200"></canvas>
    </div>

    <div class="refresh-info">
        Last updated: <span id="last-update">Never</span> | Auto-refresh every 5 seconds
    </div>

    <script>
        // Chart configuration
        const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
        const hourlyChart = new Chart(hourlyCtx, {
            type: 'bar',
            data: {
                labels: Array.from({length: 24}, (_, i) => `${i}:00`),
                datasets: [{
                    label: 'People Count',
                    data: Array(24).fill(0),
                    backgroundColor: 'rgba(76, 175, 80, 0.6)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

        const identityCtx = document.getElementById('identityChart').getContext('2d');
        const identityChart = new Chart(identityCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                    ]
                }]
            },
            options: {
                responsive: true
            }
        });

        // Update functions
        function updateMetrics() {
            fetch('/analytics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('current-occupancy').textContent = data.current_occupancy;
                    document.getElementById('total-today').textContent = data.total_people_today;
                    document.getElementById('peak-occupancy').textContent = data.peak_occupancy;
                    document.getElementById('avg-dwell').textContent = Math.round(data.average_dwell_time) + 's';
                    document.getElementById('people-in-zone').textContent = data.people_in_zone;
                    document.getElementById('active-tracks').textContent = data.current_occupancy;
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

                    // Update hourly chart
                    hourlyChart.data.datasets[0].data = Object.values(data.hourly_counts);
                    hourlyChart.update();

                    // Update identity chart
                    const identities = Object.keys(data.identity_distribution);
                    const counts = Object.values(data.identity_distribution);
                    identityChart.data.labels = identities.length > 0 ? identities : ['Unknown'];
                    identityChart.data.datasets[0].data = counts.length > 0 ? counts : [data.current_occupancy];
                    identityChart.update();
                })
                .catch(error => console.error('Error updating metrics:', error));
        }

        // Auto-refresh
        setInterval(updateMetrics, 5000);
        updateMetrics(); // Initial update
    </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/video_feed")
def video_feed():
    def generate_frames():
        while STATE.running:
            with STATE.frame_lock:
                if STATE.frame is None:
                    continue
                frame = STATE.frame.copy()

            # Encode frame
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/analytics_feed")
def analytics_feed():
    def generate_analytics_frames():
        while STATE.running:
            with STATE.frame_lock:
                if STATE.frame is None:
                    continue
                frame = STATE.frame.copy()

            # Draw analytics overlay
            frame = _draw_analytics_overlay(frame)

            # Encode frame
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return Response(
        generate_analytics_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/analytics")
def get_analytics():
    with STATE.analytics_lock:
        return jsonify(asdict(STATE.analytics))


def _draw_analytics_overlay(frame):
    """Draw analytics overlay on frame"""
    # Draw counting zone
    zone = CONFIG["COUNTING_ZONE"]
    h, w = frame.shape[:2]
    zone_x = int(zone["x"] * w)
    zone_y = int(zone["y"] * h)
    zone_w = int(zone["width"] * w)
    zone_h = int(zone["height"] * h)

    cv2.rectangle(
        frame, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), (0, 255, 255), 2
    )
    cv2.putText(
        frame,
        "Counting Zone",
        (zone_x, zone_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # Draw tracks
    with STATE.tracks_lock:
        for track in STATE.tracks.values():
            if track.positions:
                # Draw track trail
                for i in range(1, len(track.positions)):
                    pos1 = (
                        int(track.positions[i - 1][0]),
                        int(track.positions[i - 1][1]),
                    )
                    pos2 = (int(track.positions[i][0]), int(track.positions[i][1]))
                    cv2.line(frame, pos1, pos2, (0, 255, 0), 2)

                # Draw current position
                current_pos = track.positions[-1]
                center = (int(current_pos[0]), int(current_pos[1]))
                color = (0, 255, 0) if track.identity != "Unknown" else (0, 0, 255)
                cv2.circle(frame, center, 8, color, -1)

                # Draw track info
                info_text = f"ID: {track.track_id}"
                if track.identity != "Unknown":
                    info_text += f" | {track.identity}"
                info_text += f" | {track.dwell_time:.1f}s"

                cv2.putText(
                    frame,
                    info_text,
                    (center[0] + 15, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    # Draw analytics info
    with STATE.analytics_lock:
        analytics_text = [
            f"Occupancy: {STATE.analytics.current_occupancy}",
            f"In Zone: {STATE.analytics.people_in_zone}",
            f"Avg Dwell: {STATE.analytics.average_dwell_time:.1f}s",
        ]

        for i, text in enumerate(analytics_text):
            cv2.putText(
                frame,
                text,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    return frame


# =========================================================================
# MAIN EXECUTION
# =========================================================================
def main():
    print("🎯 EdgeVision People Counting System Starting...")
    print(f"📊 Dashboard will be available at http://localhost:{CONFIG['FLASK_PORT']}")

    # Start threads
    capture_thread = VideoCaptureThread()
    inference_thread = InferenceThread()
    logging_thread = LoggingThread()

    capture_thread.start()
    inference_thread.start()
    logging_thread.start()

    try:
        # Start Flask server
        app.run(host="0.0.0.0", port=CONFIG["FLASK_PORT"], debug=False, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        STATE.running = False
        capture_thread.join(timeout=2)
        inference_thread.join(timeout=2)
        logging_thread.join(timeout=2)
        print("Shutdown complete")


if __name__ == "__main__":
    main()
