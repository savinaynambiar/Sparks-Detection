import cv2
import numpy as np
import time
import threading
import queue
import os
from ultralytics import YOLO

# --- CONFIGURATION (Update these paths!) ---
YOLO_MODEL_PATH = r'C:\Sparks\spark.engine'
VIDEO_SOURCE = r'UPLine_VID1.mp4'
OUTPUT_FOLDER = r'C:\Sparks\FPSDetections'

# --- TUNING ---
ROI_PADDING = 25
YOLO_CONFIDENCE_THRESHOLD = 0.25
DISPLAY_WIDTH = 1280
MIN_SPARK_WIDTH = 2
MIN_SPARK_HEIGHT = 2
MATCHING_DOWNSCALE = 0.3
MODEL_IMGSZ = 480
SEARCH_WINDOW_MARGIN = 50 

# --- OPTIMIZATION FLAGS ---
AI_PIXEL_GATE_THRESHOLD = 230  # Skip AI if max brightness is below this
DISPLAY_SKIP_FRAMES = 2        # Update screen every 2nd frame

# --- HELPER ---
def format_timestamp(ms):
    seconds = int(ms / 1000)
    minutes = seconds // 60
    rem_sec = seconds % 60
    return f"{minutes:02d}_{rem_sec:02d}_{int(ms%1000)}"

# --- ASYNC IMAGE SAVER ---
class AsyncImageSaver:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            path, img = self.q.get()
            try:
                cv2.imwrite(path, img)
            except Exception as e:
                print(f"Error saving image: {e}")
            self.q.task_done()

    def save(self, path, img):
        self.q.put((path, img.copy()))

# --- FAST VIDEO LOADER ---
class FastVideoCapture:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.q = queue.Queue(maxsize=4)
        self.stopped = False
        self.seek_requested = False
        try:
            self.total_duration_ms = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS) * 1000
        except:
            self.total_duration_ms = 0
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            if self.seek_requested:
                time.sleep(0.01)
                continue
            if not self.q.full():
                ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.q.put((frame, ts))
            else:
                time.sleep(0.001)

    def read(self):
        if self.q.empty(): return False, None, 0
        frame, ts = self.q.get()
        return True, frame, ts

    def seek(self, seconds_relative):
        self.seek_requested = True
        current_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        target_ms = max(0, current_ms + (seconds_relative * 1000))
        if self.total_duration_ms > 0:
            target_ms = min(target_ms, self.total_duration_ms)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        with self.q.mutex:
            self.q.queue.clear()
        self.seek_requested = False
        print(f"Seeked to: {target_ms:.0f}ms")

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# --- SPARK DETECTOR ---
def detect_sparks_yolo(frame, model, conf_thresh):
    spark_boxes = []
    results = model(frame, imgsz=MODEL_IMGSZ, conf=conf_thresh, verbose=False)

    for box in results[0].boxes:
        xywh = box.xywh[0].cpu().numpy()
        x_min = int(xywh[0] - xywh[2] / 2)
        y_min = int(xywh[1] - xywh[3] / 2)
        width = int(xywh[2])
        height = int(xywh[3])
        
        if width < MIN_SPARK_WIDTH or height < MIN_SPARK_HEIGHT: continue 

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        
        spark_boxes.append((x_min, y_min, width, height))
    return spark_boxes

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    saver = AsyncImageSaver()

    print(f"Loading Engine: {YOLO_MODEL_PATH}")
    spark_model = YOLO(YOLO_MODEL_PATH, task='detect')

    print(f"Loading Video: {VIDEO_SOURCE}")
    cap = FastVideoCapture(VIDEO_SOURCE).start()
    time.sleep(1.0)
    
    ret, first_frame, _ = cap.read()
    if not ret: return
    video_h, video_w = first_frame.shape[:2]

    # --- 1. MANUAL TEMPLATE SELECTION ---
    select_scale = 1.0
    if video_w > 1280:
        select_scale = 1280 / video_w
        display_for_select = cv2.resize(first_frame, (0,0), fx=select_scale, fy=select_scale)
    else:
        display_for_select = first_frame

    roi = cv2.selectROI("Select Pantograph", display_for_select, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Pantograph")

    if roi == (0, 0, 0, 0): return

    # Scale selection back to original resolution
    x, y, w, h = roi
    x, y, w, h = int(x/select_scale), int(y/select_scale), int(w/select_scale), int(h/select_scale)

    # Create SINGLE Template
    template_full = first_frame[y:y+h, x:x+w]
    template_gray = cv2.cvtColor(template_full, cv2.COLOR_BGR2GRAY)
    temp_h, temp_w = template_gray.shape[:2]

    # Downscale Template
    small_w = int(temp_w * MATCHING_DOWNSCALE)
    small_h = int(temp_h * MATCHING_DOWNSCALE)
    small_template = cv2.resize(template_gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    print(f"Template Selected! Size: {temp_w}x{temp_h} (Scanning at: {small_w}x{small_h})")

    # Warmup
    print("Warming up GPU...")
    dummy = np.zeros((MODEL_IMGSZ, MODEL_IMGSZ, 3), dtype='uint8')
    for _ in range(5): spark_model(dummy, imgsz=MODEL_IMGSZ, verbose=False)

    display_aspect_ratio = video_h / video_w
    display_h = int(DISPLAY_WIDTH * display_aspect_ratio)
    display_size = (DISPLAY_WIDTH, display_h)

    print("🚀 STARTING OPTIMIZED DETECTION...")
    print("ℹ️  NOTE: Only frames slower than 30ms (Lag) will be printed below.")
    
    frames_in_sec = 0
    last_fps_time = time.time()
    display_fps = 0
    last_log_time = 0 
    last_loc_small = None 
    upscale_factor = 1 / MATCHING_DOWNSCALE

    while True:
        t0 = time.perf_counter()

        ret, frame, current_ts_ms = cap.read()
        if not ret:
            if cap.stopped: break
            time.sleep(0.005)
            continue

        t1 = time.perf_counter() # Read Done

        frames_in_sec += 1

        # --- OPTIMIZED TRACKING ---
        small_frame = cv2.resize(frame, (0, 0), fx=MATCHING_DOWNSCALE, fy=MATCHING_DOWNSCALE, interpolation=cv2.INTER_NEAREST)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        t2 = time.perf_counter() # Prep Done

        # Localized Search Window
        if last_loc_small is None:
            search_crop = gray_small
            search_offset_x, search_offset_y = 0, 0
        else:
            lx, ly = last_loc_small
            search_margin_scaled = int(SEARCH_WINDOW_MARGIN * MATCHING_DOWNSCALE)
            y1 = max(0, ly - search_margin_scaled)
            y2 = min(gray_small.shape[0], ly + small_h + search_margin_scaled)
            x1 = max(0, lx - search_margin_scaled)
            x2 = min(gray_small.shape[1], lx + small_w + search_margin_scaled)
            
            search_crop = gray_small[y1:y2, x1:x2]
            search_offset_x, search_offset_y = x1, y1

        # Single Template Match
        res = cv2.matchTemplate(search_crop, small_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        best_loc_small = (max_loc[0] + search_offset_x, max_loc[1] + search_offset_y)
        last_loc_small = best_loc_small

        panto_x = int(best_loc_small[0] * upscale_factor)
        panto_y = int(best_loc_small[1] * upscale_factor)
        
        roi_x1 = max(0, panto_x - ROI_PADDING)
        roi_y1 = max(0, panto_y - ROI_PADDING)
        roi_x2 = min(video_w, panto_x + temp_w + ROI_PADDING)
        roi_y2 = min(video_h, panto_y + temp_h + ROI_PADDING)
        
        t3 = time.perf_counter() # Tracking Done

        # --- INTELLIGENT DETECTION (PIXEL GATING) ---
        # --- INTELLIGENT DETECTION (SHAPE FILTER) ---
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        should_run_ai = False
        
        if roi_frame.size > 0:
            roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Threshold: Find ONLY very bright objects (Sparks or Glinting Wires)
            # Increased threshold to 245 to ignore dull reflections
            _, bright_mask = cv2.threshold(roi_gray, 245, 255, cv2.THRESH_BINARY)
            
            # 2. Check shapes of bright spots
            if cv2.countNonZero(bright_mask) > 0:
                contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    
                    # Filter A: Too small? (Noise)
                    if area < 10: continue
                    
                    # Filter B: Aspect Ratio (The Wire Killer)
                    # Wires are long/thin. Sparks are blobby (square-ish).
                    # We check if the ratio of Width/Height is extreme.
                    ratio = float(w) / h
                    
                    # If the object is roughly "square-ish" (ratio between 0.3 and 3.0), it's a spark.
                    # If it's a long line (ratio > 3.0 or < 0.3), it's a wire/bar -> SKIP.
                    if 0.25 < ratio < 4.0:
                        should_run_ai = True
                        break # Found one valid candidate, run the AI!

        if should_run_ai:
            roi_enhanced = cv2.dilate(roi_frame, np.ones((3, 3), np.uint8), iterations=1)
            spark_boxes_relative = detect_sparks_yolo(roi_enhanced, spark_model, YOLO_CONFIDENCE_THRESHOLD)
            
            for (sx, sy, sw, sh) in spark_boxes_relative:
                abs_x, abs_y = sx + roi_x1, sy + roi_y1
                cv2.rectangle(frame, (abs_x, abs_y), (abs_x + sw, abs_y + sh), (0, 0, 255), 2)
                
                # Logging
                current_sys_time = time.time()
                if (current_sys_time - last_log_time) > 1.0: 
                    time_str = format_timestamp(current_ts_ms)
                    print(f"\n🔥 SPARK DETECTED at {time_str}")
                    saver.save(f"{OUTPUT_FOLDER}/spark_{time_str}_AI_VIEW.jpg", roi_enhanced)
                    last_log_time = current_sys_time
        
        t4 = time.perf_counter() # AI Done

        # --- FPS CALCULATION ---
        curr_time = time.time()
        if curr_time - last_fps_time >= 1.0:
            display_fps = frames_in_sec
            frames_in_sec = 0
            last_fps_time = curr_time

        # --- OPTIMIZED VISUALIZATION ---
        if frames_in_sec % DISPLAY_SKIP_FRAMES == 0:
            cv2.rectangle(frame, (panto_x, panto_y), (panto_x + temp_w, panto_y + temp_h), (255, 0, 0), 2)
            cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {display_fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            timestamp_str = format_timestamp(current_ts_ms).replace('_', ':')
            cv2.putText(frame, f"Time: {timestamp_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Main View', display_frame)

        t5 = time.perf_counter() # Viz Done

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('d'): cap.seek(5)
        elif key == ord('a'): cap.seek(-5)

        # --- PRINT TIMING (FILTERED) ---
        dt_track = (t3 - t2) * 1000
        dt_ai    = (t4 - t3) * 1000
        dt_viz   = (t5 - t4) * 1000
        dt_total = (t5 - t0) * 1000

        # [MODIFIED] Only print if total time > 30ms
        if dt_total > 30.0:
             print(f"⚠️ LAG: {dt_total:.1f}ms | Track:{dt_track:.1f} | AI:{dt_ai:.1f} | Viz:{dt_viz:.1f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()