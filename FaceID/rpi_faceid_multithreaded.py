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

# =========================================================================
#  CONFIGURATION
# =========================================================================
CURRENT_MODEL = "buffalo_sc"

CONFIG = {
    "DB_PATH": f"embeddings_{CURRENT_MODEL}.pkl",
    "THRESHOLD": 0.50,
    
    # CAMERA SELECTION
    # 0 is usually the default. If you plug in both, try 0 or 2.
    "CAMERA_ID": 0,
    
    # MODEL SELECTION
    # Use 'buffalo_sc' for best performance on Pi 4 CPU
    "DETECTOR_MODEL": CURRENT_MODEL,
    
    # RESOLUTION SETTINGS
    # Stream: What you see in the browser (HD)
    "STREAM_WIDTH": 1920,
    "STREAM_HEIGHT": 1080,
    
    # Inference: What the AI processes (Low Res for Speed)
    "INFER_WIDTH": 640,
    "INFER_HEIGHT": 480,

    "CSV_LOG_PATH": "face_log.csv",
    "FLASK_PORT": 5000
}

# =========================================================================
#  SHARED STATE MANAGEMENT
# =========================================================================
class SharedState:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()
        
        self.latest_detections = []
        self.det_lock = threading.Lock()
        
        self.log_queue = queue.Queue()
        self.running = True

# =========================================================================
#  THREAD 1: VIDEO CAPTURE
# =========================================================================
class VideoCaptureThread(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state

    def run(self):
        print(f"Initializing Camera Index {CONFIG['CAMERA_ID']}...")
        cap = cv2.VideoCapture(CONFIG["CAMERA_ID"])
        
        # Performance optimization for Pi Camera / USB
        # Force MJPG mode if available to reduce USB bandwidth overhead
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Set HD Resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["STREAM_WIDTH"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["STREAM_HEIGHT"])
        
        # Verify
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera Active. Resolution: {int(actual_w)}x{int(actual_h)}")

        while self.state.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            with self.state.frame_lock:
                self.state.frame = frame.copy()
            
            # Limit capture loop to ~30 FPS to save CPU
            time.sleep(0.03)
        
        cap.release()

# =========================================================================
#  THREAD 2: AI INFERENCE
# =========================================================================
class AIInferenceThread(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.load_resources()
        
        # Calculate scaling factors to map Low-Res coordinates to High-Res stream
        self.scale_x = CONFIG["STREAM_WIDTH"] / CONFIG["INFER_WIDTH"]
        self.scale_y = CONFIG["STREAM_HEIGHT"] / CONFIG["INFER_HEIGHT"]

    def load_resources(self):
        if not os.path.exists(CONFIG["DB_PATH"]):
            print(f"Error: {CONFIG['DB_PATH']} not found.")
            exit(1)
            
        with open(CONFIG["DB_PATH"], "rb") as f:
            self.db = pickle.load(f)
        print(f"Database Loaded: {len(self.db)} identities.")

        print(f"Loading Model: {CONFIG['DETECTOR_MODEL']}")
        self.app = FaceAnalysis(name=CONFIG['DETECTOR_MODEL'], root='.')
        
        # ctx_id=-1 forces CPU mode (Required for Pi 4)
        self.app.prepare(ctx_id=-1, det_size=(640, 640)) 

    def run(self):
        print("Inference Thread Started.")
        
        while self.state.running:
            # Get latest frame safely
            with self.state.frame_lock:
                if self.state.frame is None:
                    time.sleep(0.1)
                    continue
                hd_input = self.state.frame
            
            # Downscale for processing
            small_frame = cv2.resize(hd_input, (CONFIG["INFER_WIDTH"], CONFIG["INFER_HEIGHT"]))
            
            # Run Inference
            faces = self.app.get(small_frame)
            
            results = []
            
            for face in faces:
                bbox = face.bbox.astype(int)
                kps = face.kps.astype(int)
                
                # Normalize embedding
                target_emb = face.embedding
                target_emb = target_emb / np.linalg.norm(target_emb)
                
                best_name = "Unknown"
                best_score = 0.0
                
                # Compare against database (Supports List or Single Array)
                for name, db_data in self.db.items():
                    score = 0.0
                    
                    if isinstance(db_data, list):
                        # Handle Multi-Modal (List of vectors)
                        for center in db_data:
                            curr_score = np.dot(target_emb, center.T)
                            if curr_score > score: score = curr_score
                    else:
                        # Handle Single Centroid
                        score = np.dot(target_emb, db_data.T)
                    
                    if score > best_score:
                        best_score = score
                        best_match = name
                
                # Thresholding
                final_name = best_match if best_score > CONFIG["THRESHOLD"] else "Unknown"
                color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)

                # Upscale coordinates to HD
                scaled_bbox = [
                    int(bbox[0] * self.scale_x),
                    int(bbox[1] * self.scale_y),
                    int(bbox[2] * self.scale_x),
                    int(bbox[3] * self.scale_y)
                ]
                
                scaled_kps = []
                for p in kps:
                    scaled_kps.append((int(p[0] * self.scale_x), int(p[1] * self.scale_y)))

                results.append({
                    "bbox": scaled_bbox,
                    "kps": scaled_kps,
                    "name": final_name,
                    "score": best_score,
                    "color": color
                })
                
                if final_name != "Unknown":
                    self.state.log_queue.put((final_name, best_score))

            # Update shared results
            with self.state.det_lock:
                self.state.latest_detections = results

# =========================================================================
#  THREAD 3: LOGGING
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
                
                # Log debounce (prevent spamming same person)
                last_time = self.last_log.get(name)
                if last_time is None or (now - last_time).total_seconds() > 5:
                    with open(CONFIG["CSV_LOG_PATH"], "a") as f:
                        f.write(f"{now.strftime('%H:%M:%S')},{name},{score:.2f}\n")
                    self.last_log[name] = now
                    print(f"Log: {name} ({int(score*100)}%)")
            except:
                pass

# =========================================================================
#  FLASK SERVER
# =========================================================================
app = Flask(__name__)
STATE = SharedState()

@app.route('/')
def index():
    return render_template_string("""
        <html>
        <head><title>Face Recognition Stream</title></head>
        <body style="background:black; color:white; text-align:center; font-family:sans-serif;">
            <h2>Live Feed</h2>
            <div style="border: 2px solid #333; display: inline-block;">
                <img src="/video_feed" style="max-width:100%; height:auto;">
            </div>
            <p>Resolution: 1920x1080 (HD) | Detection: 640x480</p>
        </body>
        </html>
    """)

def generate_frames():
    while STATE.running:
        with STATE.frame_lock:
            if STATE.frame is None:
                time.sleep(0.02)
                continue
            # Make copy for drawing
            display_frame = STATE.frame.copy()

        with STATE.det_lock:
            detections = STATE.latest_detections

        # Draw HD Overlays
        for d in detections:
            box = d['bbox']
            # Bounding Box
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), d['color'], 3)
            
            # Label
            label = f"{d['name']} {int(d['score']*100)}%"
            # Text Background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(display_frame, (box[0], box[1]-35), (box[0]+text_size[0]+10, box[1]), d['color'], -1)
            # Text
            cv2.putText(display_frame, label, (box[0]+5, box[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Landmarks
            for p in d['kps']:
                cv2.circle(display_frame, p, 4, (255, 255, 0), -1)

        # Encode
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    vid_t = VideoCaptureThread(STATE)
    ai_t = AIInferenceThread(STATE)
    log_t = LoggerThread(STATE)
    
    vid_t.start()
    time.sleep(2) # Allow camera warmup
    ai_t.start()
    log_t.start()
    
    print(f"Server accessible at http://0.0.0.0:{CONFIG['FLASK_PORT']}")
    app.run(host='0.0.0.0', port=CONFIG['FLASK_PORT'], debug=False, threaded=True)

if __name__ == "__main__":
    main()
