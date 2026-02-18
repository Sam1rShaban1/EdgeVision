import cv2
import numpy as np
import pickle
import os
import time
import threading
import queue
from datetime import datetime
from flask import Flask, Response, render_template_string
from insightface.app import FaceAnalysis
import subprocess

CURRENT_MODEL = "buffalo_sc"

CONFIG = {
    "DB_PATH": f"embeddings_{CURRENT_MODEL}.pkl",
    "THRESHOLD": 0.50,
    "STREAM_WIDTH": 1920,
    "STREAM_HEIGHT": 1080,
    "INFER_WIDTH": 640,
    "INFER_HEIGHT": 480,
    "CSV_LOG_PATH": "face_log.csv",
    "FLASK_PORT": 5000,
    "FRAMERATE": 25,
    "INFER_FPS": 4,
    "STREAM_FPS": 10,
}

SHOW_LANDMARKS = True


class SharedState:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()
        self.latest_detections = []
        self.det_lock = threading.Lock()
        self.log_queue = queue.Queue()
        self.running = True


STATE = SharedState()

PAGE_HTML = """<html><head><title>FaceID</title></head>
<body style='background:black;color:white;text-align:center;'>
<h2>EdgeVision Face Recognition</h2>
<img src='/video_feed' style='max-width:100%;'>
<p>Resolution: {}x{}</p>
</body></html>
""".format(CONFIG["STREAM_WIDTH"], CONFIG["STREAM_HEIGHT"])

app = Flask(__name__)


def capture_worker():
    frame_len = int(CONFIG["STREAM_WIDTH"] * CONFIG["STREAM_HEIGHT"] * 3)
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
            import io
            from PIL import Image

            img = Image.open(io.BytesIO(jpg))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            with STATE.frame_lock:
                STATE.frame = frame.copy()

    process.terminate()
    print("[INFO] Capture thread stopped.")


def inference_worker():
    print("Initializing InsightFace Model...")
    app_insight = FaceAnalysis(name=CURRENT_MODEL, root=".")

    try:
        app_insight.prepare(ctx_id=0, det_size=(640, 640))
        print("Inference Engine: GPU (CUDA) Initialized.")
    except Exception:
        print("Warning: GPU Init Failed. Falling back to CPU.")
        app_insight.prepare(ctx_id=-1, det_size=(640, 640))

    if os.path.exists(CONFIG["DB_PATH"]):
        with open(CONFIG["DB_PATH"], "rb") as f:
            db = pickle.load(f)
        print(f"Database Loaded: {len(db)} identities.")
    else:
        db = {}
        print("No identity database found.")

    last_infer = 0
    infer_interval = 1.0 / CONFIG["INFER_FPS"]
    last_seen = {}

    while STATE.running:
        now = time.time()
        if now - last_infer < infer_interval:
            time.sleep(0.01)
            continue

        with STATE.frame_lock:
            if STATE.frame is None:
                time.sleep(0.01)
                continue
            frame = STATE.frame.copy()

        faces = app_insight.get(frame)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            kps = face.kps.astype(int)
            target_emb = face.embedding / np.linalg.norm(face.embedding)

            best_name = "Unknown"
            best_score = 0
            for name, db_data in db.items():
                score = 0
                if isinstance(db_data, list):
                    for center in db_data:
                        score = max(score, np.dot(target_emb, center.T))
                else:
                    score = np.dot(target_emb, db_data.T)
                if score > best_score:
                    best_score = score
                    best_name = name

            final_name = best_name if best_score > CONFIG["THRESHOLD"] else "Unknown"
            color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)

            results.append(
                {
                    "bbox": bbox.tolist(),
                    "kps": kps.tolist(),
                    "name": final_name,
                    "score": best_score,
                    "color": color,
                }
            )

            if final_name != "Unknown":
                last = last_seen.get(final_name, 0)
                if now - last > 2:
                    STATE.log_queue.put((final_name, best_score))
                    last_seen[final_name] = now

        with STATE.det_lock:
            STATE.latest_detections = results

        last_infer = now


def logger_worker():
    if not os.path.exists(CONFIG["CSV_LOG_PATH"]):
        with open(CONFIG["CSV_LOG_PATH"], "w") as f:
            f.write("Time,Name,Confidence\n")

    last_log = {}
    while STATE.running:
        try:
            name, score = STATE.log_queue.get(timeout=1)
            now = datetime.now()
            last_time = last_log.get(name)
            if last_time is None or (now - last_time).total_seconds() > 5:
                with open(CONFIG["CSV_LOG_PATH"], "a") as f:
                    f.write(f"{now.strftime('%H:%M:%S')},{name},{score:.2f}\n")
                last_log[name] = now
                print(f"[LOG] {name} ({int(score * 100)}%)")
        except queue.Empty:
            pass


def generate_frames():
    last_frame = 0
    interval = 1.0 / CONFIG["STREAM_FPS"]

    while STATE.running:
        now = time.time()
        if now - last_frame < interval:
            time.sleep(0.005)
            continue
        last_frame = now

        with STATE.frame_lock:
            if STATE.frame is None:
                continue
            display_frame = STATE.frame.copy()

        with STATE.det_lock:
            detections = STATE.latest_detections

        for d in detections:
            bbox = d["bbox"]
            cv2.rectangle(
                display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), d["color"], 3
            )
            label = f"{d['name']} {int(d['score'] * 100)}%"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(
                display_frame,
                (bbox[0], bbox[1] - 35),
                (bbox[0] + ts[0] + 10, bbox[1]),
                d["color"],
                -1,
            )
            cv2.putText(
                display_frame,
                label,
                (bbox[0] + 5, bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            for p in d["kps"]:
                cv2.circle(display_frame, (int(p[0]), int(p[1])), 4, (255, 255, 0), -1)

        ret, buffer = cv2.imencode(
            ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60]
        )
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    t1 = threading.Thread(target=capture_worker, daemon=True)
    t2 = threading.Thread(target=inference_worker, daemon=True)
    t3 = threading.Thread(target=logger_worker, daemon=True)

    t1.start()
    time.sleep(1)
    t2.start()
    t3.start()

    print(f"[INFO] Server running at http://0.0.0.0:{CONFIG['FLASK_PORT']}")
    try:
        app.run(host="0.0.0.0", port=CONFIG["FLASK_PORT"], debug=False, threaded=True)
    finally:
        STATE.running = False
        print("[INFO] Shutdown complete.")
