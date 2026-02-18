#!/usr/bin/env python3
"""
EdgeVision People Counting & Analytics System
Uses rpicam-vid for capture, Flask for streaming (headless compatible)
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
import io
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from flask import Flask, Response, render_template_string, jsonify
from insightface.app import FaceAnalysis
import subprocess
from PIL import Image

CURRENT_MODEL = "buffalo_sc"
CONFIG = {
    "DB_PATH": f"embeddings_{CURRENT_MODEL}.pkl",
    "DETECTOR_MODEL": CURRENT_MODEL,
    "THRESHOLD": 0.50,
    "STREAM_WIDTH": 1920,
    "STREAM_HEIGHT": 1080,
    "INFER_WIDTH": 640,
    "INFER_HEIGHT": 480,
    "PEOPLE_LOG_PATH": "people_analytics.csv",
    "FLASK_PORT": 5001,
    "FRAMERATE": 25,
    "INFER_FPS": 4,
    "STREAM_FPS": 10,
    "DWELL_TIME_THRESHOLD": 30,
    "COUNTING_ZONE": {
        "x": 0.2,
        "y": 0.1,
        "width": 0.6,
        "height": 0.8,
    },
    "MAX_TRACKING_AGE": 60,
}


@dataclass
class PersonTrack:
    track_id: int
    identity: Optional[str] = None
    first_seen: datetime = None
    last_seen: datetime = None
    positions: deque = None
    dwell_time: float = 0.0
    in_counting_zone: bool = False
    crossing_count: int = 0

    def __post_init__(self):
        if self.positions is None:
            self.positions = deque(maxlen=50)
        if self.first_seen is None:
            self.first_seen = datetime.now()
        if self.last_seen is None:
            self.last_seen = datetime.now()


@dataclass
class AnalyticsData:
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


class SharedState:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()
        self.tracks = {}
        self.tracks_lock = threading.Lock()
        self.analytics = AnalyticsData()
        self.analytics_lock = threading.Lock()
        self.next_track_id = 1
        self.log_queue = queue.Queue()
        self.running = True
        self.latest_detections = []


STATE = SharedState()
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>EdgeVision People Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: #2a2a2a; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #333; }
        .metric-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
        .metric-label { font-size: 0.9em; color: #ccc; margin-top: 5px; }
        .video-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .video-box { background: #2a2a2a; padding: 10px; border-radius: 10px; text-align: center; }
        .video-box img { max-width: 100%; border-radius: 5px; }
        .chart-container { background: #2a2a2a; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .refresh-info { text-align: center; color: #888; font-size: 0.8em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>EdgeVision People Analytics</h1>
        <p>Real-time foot traffic tracking and dwell time analysis</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card"><div class="metric-value" id="current-occupancy">0</div><div class="metric-label">Current Occupancy</div></div>
        <div class="metric-card"><div class="metric-value" id="total-today">0</div><div class="metric-label">Total People Today</div></div>
        <div class="metric-card"><div class="metric-value" id="peak-occupancy">0</div><div class="metric-label">Peak Occupancy</div></div>
        <div class="metric-card"><div class="metric-value" id="avg-dwell">0s</div><div class="metric-label">Avg Dwell Time</div></div>
        <div class="metric-card"><div class="metric-value" id="people-in-zone">0</div><div class="metric-label">In Counting Zone</div></div>
        <div class="metric-card"><div class="metric-value" id="active-tracks">0</div><div class="metric-label">Active Tracks</div></div>
    </div>

    <div class="video-container">
        <div class="video-box"><h3>Live Video Feed</h3><img src="/video_feed" alt="Live feed"></div>
        <div class="video-box"><h3>Analytics View</h3><img src="/analytics_feed" alt="Analytics overlay"></div>
    </div>

    <div class="chart-container"><h3>Hourly Traffic Distribution</h3><canvas id="hourlyChart" width="400" height="200"></canvas></div>
    <div class="chart-container"><h3>Identity Distribution</h3><canvas id="identityChart" width="400" height="200"></canvas></div>

    <div class="refresh-info">Last updated: <span id="last-update">Never</span> | Auto-refresh every 5 seconds</div>

    <script>
        const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
        const hourlyChart = new Chart(hourlyCtx, { type: 'bar', data: { labels: Array.from({length: 24}, (_, i) => `${i}:00`), datasets: [{ label: 'People Count', data: Array(24).fill(0), backgroundColor: 'rgba(76, 175, 80, 0.6)', borderColor: 'rgba(76, 175, 80, 1)', borderWidth: 1 }] }, options: { responsive: true, scales: { y: { beginAtZero: true } } } });

        const identityCtx = document.getElementById('identityChart').getContext('2d');
        const identityChart = new Chart(identityCtx, { type: 'doughnut', data: { labels: [], datasets: [{ data: [], backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'] }] }, options: { responsive: true } });

        function updateMetrics() {
            fetch('/analytics').then(r => r.json()).then(data => {
                document.getElementById('current-occupancy').textContent = data.current_occupancy;
                document.getElementById('total-today').textContent = data.total_people_today;
                document.getElementById('peak-occupancy').textContent = data.peak_occupancy;
                document.getElementById('avg-dwell').textContent = Math.round(data.average_dwell_time) + 's';
                document.getElementById('people-in-zone').textContent = data.people_in_zone;
                document.getElementById('active-tracks').textContent = data.current_occupancy;
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                hourlyChart.data.datasets[0].data = Object.values(data.hourly_counts);
                hourlyChart.update();
                const identities = Object.keys(data.identity_distribution);
                const counts = Object.values(data.identity_distribution);
                identityChart.data.labels = identities.length > 0 ? identities : ['Unknown'];
                identityChart.data.datasets[0].data = counts.length > 0 ? counts : [data.current_occupancy];
                identityChart.update();
            });
        }
        setInterval(updateMetrics, 5000);
        updateMetrics();
    </script>
</body>
</html>
"""


def capture_worker():
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
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        cmd[0] = "libcamera-vid"
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

    bytes_data = b""
    while STATE.running:
        bytes_data += process.stdout.read(8192)
        a = bytes_data.find(b"\xff\xd8")
        b = bytes_data.find(b"\xff\xd9")
        if a != -1 and b != -1:
            jpg = bytes_data[a : b + 2]
            bytes_data = bytes_data[b + 2 :]
            img = Image.open(io.BytesIO(jpg))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            with STATE.frame_lock:
                STATE.frame = frame.copy()

    process.terminate()
    print("[INFO] Capture thread stopped.")


def inference_worker():
    print("Initializing People Counter...")
    app_insight = FaceAnalysis(name=CONFIG["DETECTOR_MODEL"], root=".")

    try:
        app_insight.prepare(ctx_id=0, det_size=(640, 640))
        print("Inference Engine: GPU (CUDA) Initialized.")
    except Exception:
        print("Warning: GPU Init Failed. Falling back to CPU.")
        app_insight.prepare(ctx_id=-1, det_size=(640, 640))

    if os.path.exists(CONFIG["DB_PATH"]):
        with open(CONFIG["DB_PATH"], "rb") as f:
            db = pickle.load(f)
        print(f"Loaded {len(db)} identities")
    else:
        db = {}
        print("No identity database found")

    last_infer = 0
    infer_interval = 1.0 / CONFIG["INFER_FPS"]
    tracks = {}
    next_track_id = 1

    while STATE.running:
        current_time = time.time()
        if current_time - last_infer < infer_interval:
            time.sleep(0.01)
            continue

        with STATE.frame_lock:
            if STATE.frame is None:
                time.sleep(0.01)
                continue
            frame = STATE.frame.copy()

        faces = app_insight.get(frame)
        detections = []

        for face in faces:
            bbox = face.bbox.astype(int)
            target_emb = face.embedding / np.linalg.norm(face.embedding)

            identity = "Unknown"
            best_score = 0
            if len(db) > 0:
                for name, db_data in db.items():
                    score = 0.0
                    if isinstance(db_data, list):
                        for center in db_data:
                            curr_score = np.dot(target_emb, center.T)
                            if curr_score > score:
                                score = curr_score
                    else:
                        score = np.dot(target_emb, db_data.T)
                    if score > best_score:
                        best_score = score
                        identity = name if score > CONFIG["THRESHOLD"] else "Unknown"

            detections.append(
                {"bbox": bbox.tolist(), "identity": identity, "confidence": best_score}
            )

        current_dt = datetime.now()
        new_tracks = {}

        for det in detections:
            bbox = det["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            best_match_id = None
            best_distance = float("inf")

            for track_id, track in tracks.items():
                if track.positions:
                    last_pos = track.positions[-1]
                    distance = np.sqrt(
                        (center_x - last_pos[0]) ** 2 + (center_y - last_pos[1]) ** 2
                    )
                    if distance < best_distance and distance < 100:
                        best_distance = distance
                        best_match_id = track_id

            if best_match_id is not None:
                track = tracks[best_match_id]
                track.positions.append((center_x, center_y, current_dt))
                track.last_seen = current_dt
                track.dwell_time = (current_dt - track.first_seen).total_seconds()
                if det.get("identity") and det["identity"] != "Unknown":
                    track.identity = det["identity"]
                new_tracks[best_match_id] = track
            else:
                track = PersonTrack(
                    track_id=next_track_id,
                    identity=det.get("identity"),
                    first_seen=current_dt,
                    last_seen=current_dt,
                )
                track.positions.append((center_x, center_y, current_dt))
                new_tracks[next_track_id] = track
                STATE.log_queue.put(
                    {
                        "event": "person_entered",
                        "track_id": track.track_id,
                        "identity": track.identity,
                        "timestamp": current_dt.isoformat(),
                    }
                )
                next_track_id += 1

        for track_id in list(tracks.keys()):
            if track_id not in new_tracks:
                track = tracks[track_id]
                age = (current_dt - track.last_seen).total_seconds()
                if age > CONFIG["MAX_TRACKING_AGE"]:
                    STATE.log_queue.put(
                        {
                            "event": "person_exited",
                            "track_id": track.track_id,
                            "identity": track.identity,
                            "dwell_time": track.dwell_time,
                            "timestamp": current_dt.isoformat(),
                        }
                    )
                    del tracks[track_id]
                else:
                    new_tracks[track_id] = track
            else:
                new_tracks[track_id] = tracks[track_id]

        tracks = new_tracks

        with STATE.analytics_lock:
            STATE.analytics.current_occupancy = len(tracks)
            STATE.analytics.peak_occupancy = max(
                STATE.analytics.peak_occupancy, STATE.analytics.current_occupancy
            )
            zone_people = 0
            zone = CONFIG["COUNTING_ZONE"]
            for track in tracks.values():
                if track.positions:
                    pos = track.positions[-1]
                    if (
                        zone["x"] * CONFIG["STREAM_WIDTH"]
                        <= pos[0]
                        <= (zone["x"] + zone["width"]) * CONFIG["STREAM_WIDTH"]
                        and zone["y"] * CONFIG["STREAM_HEIGHT"]
                        <= pos[1]
                        <= (zone["y"] + zone["height"]) * CONFIG["STREAM_HEIGHT"]
                    ):
                        zone_people += 1
            STATE.analytics.people_in_zone = zone_people
            if tracks:
                STATE.analytics.average_dwell_time = np.mean(
                    [t.dwell_time for t in tracks.values()]
                )
            STATE.analytics.hourly_counts[current_dt.hour] = (
                STATE.analytics.current_occupancy
            )
            STATE.analytics.identity_distribution.clear()
            for track in tracks.values():
                if track.identity:
                    STATE.analytics.identity_distribution[track.identity] += 1

        with STATE.tracks_lock:
            STATE.tracks = tracks
            STATE.latest_detections = detections

        last_infer = current_time


def logger_worker():
    csv_file = CONFIG["PEOPLE_LOG_PATH"]
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


def draw_analytics_overlay(frame):
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

    with STATE.tracks_lock:
        for track in STATE.tracks.values():
            if track.positions:
                for i in range(1, len(track.positions)):
                    pos1 = (
                        int(track.positions[i - 1][0]),
                        int(track.positions[i - 1][1]),
                    )
                    pos2 = (int(track.positions[i][0]), int(track.positions[i][1]))
                    cv2.line(frame, pos1, pos2, (0, 255, 0), 2)
                current_pos = track.positions[-1]
                center = (int(current_pos[0]), int(current_pos[1]))
                color = (0, 255, 0) if track.identity != "Unknown" else (0, 0, 255)
                cv2.circle(frame, center, 8, color, -1)
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
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
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
            frame = draw_analytics_overlay(frame)
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    return Response(
        generate_analytics_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/analytics")
def get_analytics():
    with STATE.analytics_lock:
        return jsonify(asdict(STATE.analytics))


if __name__ == "__main__":
    print("EdgeVision People Counting System Starting...")
    print(f"Dashboard will be available at http://localhost:{CONFIG['FLASK_PORT']}")

    t1 = threading.Thread(target=capture_worker, daemon=True)
    t2 = threading.Thread(target=inference_worker, daemon=True)
    t3 = threading.Thread(target=logger_worker, daemon=True)

    t1.start()
    time.sleep(1)
    t2.start()
    t3.start()

    try:
        app.run(host="0.0.0.0", port=CONFIG["FLASK_PORT"], debug=False, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        STATE.running = False
        print("Shutdown complete")
