import cv2
import numpy as np
import pickle
import os
import time
from insightface.app import FaceAnalysis

# =========================================================================
#  CONFIGURATION
# =========================================================================
CURRENT_MODEL = "buffalo_s"  # Change to "buffalo_s" for Pi

CONFIG = {
    # Use f-string to inject the variable
    "DB_PATH": f"embeddings_{CURRENT_MODEL}.pkl",
    
    "THRESHOLD": 0.50,        
    "CAMERA_ID": 0,           
    
    "DETECTOR_MODEL": CURRENT_MODEL, 
    
    "FRAME_WIDTH": 1080,
    "FRAME_HEIGHT": 720
}

# Global state for landmark toggling
SHOW_LANDMARKS = True 

class FaceIDRunner:
    def __init__(self):
        print("System Booting...")
        
        # 1. Validation
        if not os.path.exists(CONFIG["DB_PATH"]):
            print(f"Error: Database file '{CONFIG['DB_PATH']}' not found.")
            exit(1)
            
        # 2. Load Database
        try:
            with open(CONFIG["DB_PATH"], "rb") as f:
                self.db = pickle.load(f)
            print(f"Database Loaded: {len(self.db)} identities.")
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            exit(1)

        # 3. Initialize Engine
        print(f"Initializing InsightFace Model: {CONFIG['DETECTOR_MODEL']}...")
        self.app = FaceAnalysis(name=CONFIG['DETECTOR_MODEL'], root='.')
        
        # Execution Provider Check (Auto-detect GPU)
        try:
            # ctx_id=0 attempts to use GPU. 
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("Inference Engine: GPU (CUDA) Initialized.")
        except Exception:
            # Fallback to CPU if CUDA libraries are missing or incompatible
            print("Warning: GPU Init Failed. Falling back to CPU.")
            self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def run(self):
        global SHOW_LANDMARKS 
        
        cap = cv2.VideoCapture(CONFIG["CAMERA_ID"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])

        if not cap.isOpened():
            print("Error: Camera could not be opened. Check connection or permissions.")
            return

        print(f"Camera Active ({CONFIG['FRAME_WIDTH']}x{CONFIG['FRAME_HEIGHT']}).")
        print("Controls: 'D' to toggle landmarks, 'Q' to quit.")

        prev_time = 0
        cv2.namedWindow("FaceID")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # Detection Inference
            faces = self.app.get(frame)

            for face in faces:
                bbox = face.bbox.astype(int)
                
                # Embedding Normalization (Critical for Cosine Similarity)
                target_emb = face.embedding
                target_emb = target_emb / np.linalg.norm(target_emb)

                best_name = "Unknown"
                best_score = 0.0

                # --- Robust Comparison Logic ---
                for name, db_data in self.db.items():
                    score = 0.0
                    
                    if isinstance(db_data, list):
                        # Multi-modal: List of vectors (from clustered generation)
                        for center in db_data:
                            curr_score = np.dot(target_emb, center.T)
                            if curr_score > score:
                                score = curr_score
                    else:
                        # Single Centroid: Numpy Array (from simple generation)
                        score = np.dot(target_emb, db_data.T)

                    if score > best_score:
                        best_score = score
                        best_match = name

                # --- Result Logic ---
                if best_score > CONFIG["THRESHOLD"]:
                    color = (0, 255, 0) # Green
                    label = f"{best_match} {int(best_score*100)}%"
                else:
                    color = (0, 0, 255) # Red
                    label = f"Unknown {int(best_score*100)}%"

                # Draw UI
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Text Background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display, (bbox[0], bbox[1] - 25), (bbox[0] + text_size[0] + 10, bbox[1]), color, -1)
                
                cv2.putText(display, label, (bbox[0] + 5, bbox[1] - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # --- Robust Landmark Rendering ---
                if SHOW_LANDMARKS:
                    # 1. Try to get high-detail 2D landmarks (106 points)
                    # getattr is safe; it returns None if the attribute doesn't exist
                    landmarks = getattr(face, 'landmark_2d_106', None)
                    
                    if landmarks is not None:
                        # Render 106 points (Buffalo_L style)
                        for p in landmarks:
                            cv2.circle(display, (int(p[0]), int(p[1])), 1, (0, 255, 255), -1)
                    else:
                        # Fallback to standard 5 keypoints (Buffalo_S style)
                        # face.kps is guaranteed to exist if a face was detected
                        kps = face.kps.astype(int)
                        for p in kps:
                            cv2.circle(display, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            cv2.putText(display, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("FaceID", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                SHOW_LANDMARKS = not SHOW_LANDMARKS 

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceIDRunner()
    app.run()