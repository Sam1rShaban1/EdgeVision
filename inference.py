import cv2
import threading
import queue
import time
import pytesseract
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# ================= CONFIGURATION =================
# Path to your INT8 NCNN model folder
MODEL_PATH = "yolov8n_ncnn_int8" 

# File logging
CSV_FILE = "plate_log.csv"

# Camera & Inference Settings
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
INFERENCE_SIZE = 640   # Fixed 640x640
CONF_THRESHOLD = 0.50  

# OCR Settings (Alphanumeric whitelist)
OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ================= SHARED RESOURCES =================
frame_queue = queue.Queue(maxsize=1)
ocr_queue = queue.Queue(maxsize=1)
display_queue = queue.Queue(maxsize=1)

stop_event = threading.Event()

# Shared variable for display
latest_plate_text = "Scanning..."
text_lock = threading.Lock()

# Globals for Logging Logic
last_logged_text = ""
last_logged_time = 0

# ================= SETUP CSV =================
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Detected Text"])
    print(f"[INFO] Created new log file: {CSV_FILE}")

# ================= THREAD 1: CAPTURE =================
def capture_worker():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Keep queue fresh
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

    cap.release()

# ================= THREAD 2: YOLO + VISUALIZATION =================
def yolo_worker():
    model = YOLO(MODEL_PATH, task='detect')
    box_color = (0, 255, 0) # Green

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # 1. Inference
        results = model(frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, verbose=False)
        
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Boundary checks
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # 2. Crop for OCR (only if queue is empty to prevent lag)
            if not ocr_queue.full():
                plate_crop = frame[y1:y2, x1:x2]
                ocr_queue.put(plate_crop)

            # 3. Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

            # 4. Draw the OCR Text ON the box
            with text_lock:
                current_text = latest_plate_text
            
            # Text Background
            label = f"{current_text}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), box_color, -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Send to display
        if not display_queue.empty():
            try:
                display_queue.get_nowait()
            except queue.Empty:
                pass
        display_queue.put(frame)

# ================= THREAD 3: OCR + CSV LOGGING =================
def ocr_worker():
    global latest_plate_text, last_logged_text, last_logged_time
    
    while not stop_event.is_set():
        try:
            plate_img = ocr_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # --- Pre-processing ---
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # Upscale 2x for Tesseract
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # --- Inference ---
        text = pytesseract.image_to_string(thresh, config=OCR_CONFIG)
        clean_text = "".join(text.split()).strip()
        
        # --- Filtering & Logging ---
        if len(clean_text) >= 3:
            # Update Display Variable
            with text_lock:
                latest_plate_text = clean_text

            # CSV Logging Logic
            current_time = time.time()
            
            # Log if:
            # 1. The text is different from the last one
            # OR
            # 2. It's the same text, but 5 seconds have passed (re-log same car after delay)
            if (clean_text != last_logged_text) or (current_time - last_logged_time > 5.0):
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, clean_text])
                
                print(f"[LOGGED] {timestamp} - {clean_text}")
                
                last_logged_text = clean_text
                last_logged_time = current_time

# ================= MAIN =================
if __name__ == "__main__":
    t1 = threading.Thread(target=capture_worker, daemon=True)
    t2 = threading.Thread(target=yolo_worker, daemon=True)
    t3 = threading.Thread(target=ocr_worker, daemon=True)

    t1.start()
    time.sleep(1.0)
    t2.start()
    t3.start()

    print(f"System Running. Logs saved to {CSV_FILE}. Press 'q' to quit.")

    try:
        while True:
            try:
                frame = display_queue.get(timeout=0.1)
                cv2.imshow("Pi 4 LPR - YOLO INT8", frame)
            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        t1.join()
        t2.join()
        t3.join()
        cv2.destroyAllWindows()
