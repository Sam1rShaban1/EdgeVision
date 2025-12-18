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

# =========================================================================
# CONFIG
# =========================================================================
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
    "FRAMERATE": 25
}

# =========================================================================
# SHARED STATE
# =========================================================================
class SharedState:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()
        self.latest_detections = []
        self.det_lock = threading.Lock()
        self.log_queue = queue.Queue()
        self.running = True

STATE = SharedState()

# =========================================================================
# VIDEO CAPTURE THREAD
# =========================================================================
# THREAD 1: VIDEO CAPTURE USING LIBCAMERA
class VideoCaptureThread(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.frame_len = int(CONFIG["STREAM_WIDTH"] * CONFIG["STREAM_HEIGHT"] * 1.5)  # YUV420

    def run(self):
        cmd = [
            "libcamera-vid",
            "--nopreview",
            "--width", str(CONFIG["STREAM_WIDTH"]),
            "--height", str(CONFIG["STREAM_HEIGHT"]),
            "--framerate", "25",
            "--codec", "yuv420",
            "-o", "-"
        ]
        print("[INFO] Starting HQ Camera subprocess...")
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**7)
        except FileNotFoundError:
            print("[ERROR] libcamera-vid not found. Make sure it's installed and on PATH.")
            return

        while self.state.running:
            raw_data = process.stdout.read(self.frame_len)
            if len(raw_data) != self.frame_len:
                time.sleep(0.01)
                continue

            yuv_image = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                (int(CONFIG["STREAM_HEIGHT"] * 1.5), CONFIG["STREAM_WIDTH"])
            )
            frame = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)

            with self.state.frame_lock:
                self.state.frame = frame

        process.terminate()
        print("[INFO] Video capture stopped.")


# =========================================================================
# AI INFERENCE THREAD
# =========================================================================
class AIInferenceThread(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.load_resources()
        self.scale_x = CONFIG["STREAM_WIDTH"] / CONFIG["INFER_WIDTH"]
        self.scale_y = CONFIG["STREAM_HEIGHT"] / CONFIG["INFER_HEIGHT"]

    def load_resources(self):
        if not os.path.exists(CONFIG["DB_PATH"]):
            print(f"Database {CONFIG['DB_PATH']} not found!")
            exit(1)
        with open(CONFIG["DB_PATH"], "rb") as f:
            self.db = pickle.load(f)
        print(f"[INFO] Loaded {len(self.db)} identities")

        print(f"[INFO] Loading InsightFace model {CURRENT_MODEL}...")
        self.app = FaceAnalysis(name=CURRENT_MODEL, root='.')
        self.app.prepare(ctx_id=-1, det_size=(640,640))
        print("[INFO] Model ready.")

    def run(self):
        while self.state.running:
            with self.state.frame_lock:
                if self.state.frame is None:
                    time.sleep(0.05)
                    continue
                hd_frame = self.state.frame.copy()

            small_frame = cv2.resize(hd_frame, (CONFIG["INFER_WIDTH"], CONFIG["INFER_HEIGHT"]))
            faces = self.app.get(small_frame)
            results = []

            for face in faces:
                bbox = face.bbox.astype(int)
                kps = face.kps.astype(int)
                target_emb = face.embedding / np.linalg.norm(face.embedding)

                best_name = "Unknown"
                best_score = 0
                for name, db_data in self.db.items():
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
                color = (0,255,0) if final_name!="Unknown" else (0,0,255)

                scaled_bbox = [
                    int(bbox[0]*self.scale_x),
                    int(bbox[1]*self.scale_y),
                    int(bbox[2]*self.scale_x),
                    int(bbox[3]*self.scale_y)
                ]
                scaled_kps = [(int(p[0]*self.scale_x), int(p[1]*self.scale_y)) for p in kps]

                results.append({
                    "bbox": scaled_bbox,
                    "kps": scaled_kps,
                    "name": final_name,
                    "score": best_score,
                    "color": color
                })

                if final_name != "Unknown":
                    self.state.log_queue.put((final_name, best_score))

            with self.state.det_lock:
                self.state.latest_detections = results

# =========================================================================
# LOGGER THREAD
# =========================================================================
class LoggerThread(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.last_log = {}

    def run(self):
        if not os.path.exists(CONFIG["CSV_LOG_PATH"]):
            with open(CONFIG["CSV_LOG_PATH"], "w") as f:
                f.write("Time,Name,Confidence\n")
        while self.state.running:
            try:
                name, score = self.state.log_queue.get(timeout=1)
                now = datetime.now()
                last_time = self.last_log.get(name)
                if last_time is None or (now-last_time).total_seconds()>5:
                    with open(CONFIG["CSV_LOG_PATH"], "a") as f:
                        f.write(f"{now.strftime('%H:%M:%S')},{name},{score:.2f}\n")
                    self.last_log[name] = now
                    print(f"[LOG] {name} ({int(score*100)}%)")
            except queue.Empty:
                pass

# =========================================================================
# FLASK SERVER
# =========================================================================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head><title>Face Recognition Stream</title></head>
    <body style="background:black; color:white; text-align:center;">
        <h2>Live Face Stream</h2>
        <img src="/video_feed" style="max-width:100%;">
        <p>Resolution: {{w}}x{{h}}</p>
    </body>
    </html>
    """, w=CONFIG["STREAM_WIDTH"], h=CONFIG["STREAM_HEIGHT"])

def generate_frames():
    while STATE.running:
        with STATE.frame_lock:
            if STATE.frame is None:
                time.sleep(0.02)
                continue
            display_frame = STATE.frame.copy()

        with STATE.det_lock:
            detections = STATE.latest_detections

        for d in detections:
            bbox = d['bbox']
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), d['color'], 3)
            label = f"{d['name']} {int(d['score']*100)}%"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(display_frame, (bbox[0], bbox[1]-35), (bbox[0]+ts[0]+10, bbox[1]), d['color'], -1)
            cv2.putText(display_frame, label, (bbox[0]+5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            for p in d['kps']:
                cv2.circle(display_frame,p,4,(255,255,0),-1)

        ret, buffer = cv2.imencode('.jpg', display_frame,[cv2.IMWRITE_JPEG_QUALITY,60])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================================================================
# MAIN
# =========================================================================
def main():
    vid_t = VideoCaptureThread(STATE)
    ai_t = AIInferenceThread(STATE)
    log_t = LoggerThread(STATE)

    vid_t.start()
    time.sleep(1)  # camera warmup
    ai_t.start()
    log_t.start()

    print(f"[INFO] Server running at http://0.0.0.0:{CONFIG['FLASK_PORT']}")
    app.run(host='0.0.0.0', port=CONFIG['FLASK_PORT'], debug=False, threaded=True)

if __name__=="__main__":
    main()
