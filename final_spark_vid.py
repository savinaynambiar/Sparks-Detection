import cv2
import numpy as np
import time
import threading
import queue
import os
import sys
from ultralytics import YOLO

# --- CONFIGURATION ---
YOLO_MODEL_PATH = r'C:\Sparks\spark.engine'
VIDEO_SOURCE = r'WIN_20250404_14_55_56_Pro.mp4'
OUTPUT_FOLDER = r'C:\Sparks\VIDDetections'

# --- TUNING ---
ROI_PADDING = 25
YOLO_CONFIDENCE_THRESHOLD = 0.35
# Display width is used for the selection window only now
DISPLAY_WIDTH = 1280 
MIN_SPARK_WIDTH = 2
MIN_SPARK_HEIGHT = 2
MATCHING_DOWNSCALE = 0.3
MODEL_IMGSZ = 480
SEARCH_WINDOW_MARGIN = 50 

# --- INTENSITY FILTER ---
# "Cloud Killer" Contrast Ratio
MIN_CONTRAST_RATIO = 1.32
INTENSITY_MIN_PEAK = 250 

# --- OPTIMIZATION FLAGS ---
AI_PIXEL_GATE_THRESHOLD = 230 
DISPLAY_SKIP_FRAMES = 2 # Affects FPS calculation frequency

# --- HELPER ---
def format_timestamp(ms):
    seconds = int(ms / 1000)
    minutes = seconds // 60
    rem_sec = seconds % 60
    return f"{minutes:02d}_{rem_sec:02d}_{int(ms%1000)}"

class AsyncImageSaver:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    def _worker(self):
        while True:
            path, img = self.q.get()
            try: cv2.imwrite(path, img)
            except: pass
            self.q.task_done()
    def save(self, path, img):
        self.q.put((path, img.copy()))

class FastVideoCapture:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.q = queue.Queue(maxsize=4)
        self.stopped = False
        self.seek_requested = False
        try: self.total_duration_ms = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS) * 1000
        except: self.total_duration_ms = 0
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
        pass # Seeking removed for headless mode
    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

def detect_sparks_yolo(frame, model, conf_thresh):
    spark_boxes = []
    results = model(frame, imgsz=MODEL_IMGSZ, conf=conf_thresh, verbose=False)
    for box in results[0].boxes:
        xywh = box.xywh[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        x_min = int(xywh[0] - xywh[2] / 2)
        y_min = int(xywh[1] - xywh[3] / 2)
        width = int(xywh[2])
        height = int(xywh[3])
        if width < MIN_SPARK_WIDTH or height < MIN_SPARK_HEIGHT: continue 
        spark_boxes.append((max(0, x_min), max(0, y_min), width, height, conf))
    return spark_boxes

def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    saver = AsyncImageSaver()
    print(f"Loading Engine: {YOLO_MODEL_PATH}")
    spark_model = YOLO(YOLO_MODEL_PATH, task='detect')
    print(f"Loading Video: {VIDEO_SOURCE}")
    cap = FastVideoCapture(VIDEO_SOURCE).start()
    time.sleep(1.0)
    
    ret, first_frame, _ = cap.read()
    if not ret: return
    video_h, video_w = first_frame.shape[:2]

    # --- INITIAL SELECTION (Window will appear ONCE then vanish) ---
    select_scale = 1.0
    if video_w > 1280:
        select_scale = 1280 / video_w
        display_for_select = cv2.resize(first_frame, (0,0), fx=select_scale, fy=select_scale)
    else: display_for_select = first_frame
    
    print("\n--- INSTRUCTIONS ---")
    print("1. Draw box around Pantograph.")
    print("2. Press SPACE.")
    print("3. Window will close and processing will run in background.")
    
    roi = cv2.selectROI("Select Pantograph", display_for_select, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Pantograph") # <--- Closes the window immediately
    
    if roi == (0, 0, 0, 0): return
    x, y, w, h = roi
    x, y, w, h = int(x/select_scale), int(y/select_scale), int(w/select_scale), int(h/select_scale)

    template_full = first_frame[y:y+h, x:x+w]
    template_gray = cv2.cvtColor(template_full, cv2.COLOR_BGR2GRAY)
    temp_h, temp_w = template_gray.shape[:2]
    small_w = int(temp_w * MATCHING_DOWNSCALE)
    small_h = int(temp_h * MATCHING_DOWNSCALE)
    small_template = cv2.resize(template_gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    print(f"Template Selected! Size: {temp_w}x{temp_h}")
    print("🚀 STARTING HEADLESS DETECTION (Ctrl+C to Stop)...")
    
    frames_in_sec = 0
    last_fps_time = time.time()
    display_fps = 0
    last_log_time = 0 
    last_loc_small = None 
    upscale_factor = 1 / MATCHING_DOWNSCALE

    try:
        while True:
            ret, frame, current_ts_ms = cap.read()
            if not ret:
                if cap.stopped: 
                    print("\nVideo Finished.")
                    break
                time.sleep(0.005)
                continue

            frames_in_sec += 1

            # Track
            small_frame = cv2.resize(frame, (0, 0), fx=MATCHING_DOWNSCALE, fy=MATCHING_DOWNSCALE, interpolation=cv2.INTER_NEAREST)
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
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

            res = cv2.matchTemplate(search_crop, small_template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            best_loc_small = (max_loc[0] + search_offset_x, max_loc[1] + search_offset_y)
            last_loc_small = best_loc_small

            panto_x = int(best_loc_small[0] * upscale_factor)
            panto_y = int(best_loc_small[1] * upscale_factor)
            roi_x1 = max(0, panto_x - ROI_PADDING)
            roi_y1 = max(0, panto_y - ROI_PADDING)
            roi_x2 = min(video_w, panto_x + temp_w + ROI_PADDING)
            roi_y2 = min(video_h, panto_y + temp_h + ROI_PADDING)
            
            # Gates
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            should_run_ai = False
            if roi_frame.size > 0:
                roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                # Lower pre-gate to allow candidates for contrast check
                _, bright_mask = cv2.threshold(roi_gray, AI_PIXEL_GATE_THRESHOLD, 255, cv2.THRESH_BINARY)
                
                if cv2.countNonZero(bright_mask) > 0:
                    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = w * h
                        if area < 10: continue
                        ratio = float(w) / h
                        if 0.25 < ratio < 4.0:
                            should_run_ai = True
                            break

            if should_run_ai:
                roi_enhanced = cv2.dilate(roi_frame, np.ones((3, 3), np.uint8), iterations=1)
                spark_boxes_relative = detect_sparks_yolo(roi_enhanced, spark_model, YOLO_CONFIDENCE_THRESHOLD)
                
                for (sx, sy, sw, sh, conf) in spark_boxes_relative:
                    # --- INTENSITY & CONTRAST CHECK ---
                    spark_crop = roi_gray[sy:sy+sh, sx:sx+sw]
                    if spark_crop.size == 0: continue
                    
                    peak_intensity = np.max(spark_crop)
                    avg_intensity = np.mean(spark_crop)
                    
                    if avg_intensity > 0:
                        contrast_ratio = peak_intensity / avg_intensity
                    else:
                        contrast_ratio = 0

                    is_valid = True
                    if peak_intensity < INTENSITY_MIN_PEAK: is_valid = False
                    elif contrast_ratio < MIN_CONTRAST_RATIO: is_valid = False

                    if not is_valid: continue 

                    # --- CONFIRMED & SAVE ---
                    current_sys_time = time.time()
                    if (current_sys_time - last_log_time) > 1.0: 
                        time_str = format_timestamp(current_ts_ms)
                        print(f"\n🔥 SPARK DETECTED | Time: {time_str} | Peak: {peak_intensity} | Ratio: {contrast_ratio:.2f}")
                        
                        # Save Annotated Image (Red Box + Text)
                        roi_visual = roi_enhanced.copy() # Make copy
                        cv2.rectangle(roi_visual, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                        cv2.putText(roi_visual, f"R:{contrast_ratio:.1f}", (sx, sy - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        saver.save(f"{OUTPUT_FOLDER}/spark_{time_str}_AI_VIEW.jpg", roi_visual)
                        last_log_time = time.time()
            
            # --- FPS CALCULATION & PRINTING ---
            curr_time = time.time()
            if curr_time - last_fps_time >= 1.0:
                display_fps = frames_in_sec
                # Update line in terminal without creating new lines
                sys.stdout.write(f"\rCurrent Speed: {display_fps} FPS   ")
                sys.stdout.flush()
                
                frames_in_sec = 0
                last_fps_time = curr_time

            # NO WINDOW DISPLAY HERE (Headless)

    except KeyboardInterrupt:
        print("\nStopping...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
