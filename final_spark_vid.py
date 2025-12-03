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
VIDEO_SOURCE = r'VID3.mp4'
OUTPUT_FOLDER = r'C:\Sparks\HybridDetections'

# --- TUNING ---
ROI_PADDING = 100 
YOLO_CONFIDENCE_THRESHOLD = 0.30 
DISPLAY_WIDTH = 640

# --- BRIGHTNESS FILTER ---
SPARK_BRIGHTNESS_THRESHOLD = 250 
AI_TRIGGER_THRESHOLD = SPARK_BRIGHTNESS_THRESHOLD - 50 

# --- PERFORMANCE & TRACKING ---
MATCHING_DOWNSCALE = 0.3
TRACKING_SKIP_FRAMES = 3    
TRACKING_CONFIDENCE_THRESHOLD = 0.4

# --- SIZE FILTERS ---
MIN_SPARK_WIDTH = 2
MIN_SPARK_HEIGHT = 2
MAX_SPARK_WIDTH = 300   
MAX_SPARK_HEIGHT = 300 

# --- MASKING ---
MASK_EXPANSION = 5 

MODEL_IMGSZ = 480
SEARCH_WINDOW_MARGIN = 80

# --- HELPER ---
def format_timestamp(ms):
    seconds = int(ms / 1000)
    minutes = seconds // 60
    rem_sec = seconds % 60
    return f"{minutes:02d}_{rem_sec:02d}_{int(ms%1000)}"

# ==========================================
#      PART 1: ROTATED BOX TOOL
# ==========================================
points = []
current_ratio = 0.20

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            return

def calculate_rotated_box(p1, p2, ratio):
    p1 = np.array(p1); p2 = np.array(p2)
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length == 0: return np.array([p1, p1, p1, p1]) 
    unit_vec = vec / length
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])
    padding = length * ratio
    c1 = p1 + (perp_vec * padding); c2 = p2 + (perp_vec * padding)
    c3 = p2 - (perp_vec * padding); c4 = p1 - (perp_vec * padding)
    return np.array([c1, c2, c3, c4], dtype=np.int32)

def get_interactive_roi(frame, scale=1.0):
    global points, current_ratio
    points = []
    current_ratio = 0.20
    
    display_img = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    display_copy = display_img.copy()
    
    cv2.namedWindow('Setup')
    cv2.setMouseCallback('Setup', click_event, {'img': display_img})
    
    print("\n--- SETUP INSTRUCTIONS ---")
    print("1. Click LEFT end of joint.")
    print("2. Click RIGHT end of joint.")
    print("3. Press 'S' to SHRINK box.")
    print("4. Press SPACE to confirm.")
    print("   (Detections will be restricted to the GREEN ZONE above the red line)")
    
    while True:
        temp_display = display_copy.copy()
        for p in points: cv2.circle(temp_display, p, 5, (0, 0, 255), -1)
        
        if len(points) == 2:
            # Draw the Red Line (The Spine)
            cv2.line(temp_display, points[0], points[1], (0, 0, 255), 2)
            
            # Calculate Full Box
            full_box = calculate_rotated_box(points[0], points[1], current_ratio)
            c1, c2, c3, c4 = full_box

            # --- LOGIC TO FIND TOP HALF (Sky Side) ---
            # In OpenCV, smaller Y is "higher" in the image
            avg_y_12 = (c1[1] + c2[1]) / 2
            avg_y_34 = (c3[1] + c4[1]) / 2

            p1_arr = np.array(points[0])
            p2_arr = np.array(points[1])

            if avg_y_12 < avg_y_34:
                # c1, c2 is the Top/Sky side
                half_poly = np.array([c1, c2, p2_arr, p1_arr], dtype=np.int32)
            else:
                # c3, c4 is the Top/Sky side
                half_poly = np.array([c4, c3, p2_arr, p1_arr], dtype=np.int32)

            # Draw Full Box Outline (Blue) - Shows what is being TRACKED
            cv2.polylines(temp_display, [full_box], True, (255, 0, 0), 2)
            
            # Draw Active Zone (Filled Green transparently) - Shows where DETECTION happens
            overlay = temp_display.copy()
            cv2.fillPoly(overlay, [half_poly], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, temp_display, 0.7, 0, temp_display)

            cv2.putText(temp_display, "Active Zone (Green)", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(temp_display, "Press SPACE to start", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Setup', temp_display)
        key = cv2.waitKey(20) & 0xFF
        if key == 32 and len(points) == 2: break
        elif key == ord('r'): points = []
        elif key == ord('w'): current_ratio += 0.02
        elif key == ord('s'): current_ratio = max(0.05, current_ratio - 0.02)

    cv2.destroyWindow('Setup')
    
    p1 = (int(points[0][0]/scale), int(points[0][1]/scale))
    p2 = (int(points[1][0]/scale), int(points[1][1]/scale))
    
    # 1. Get the FULL box for cropping (Tracking needs the whole structure)
    full_poly_points = calculate_rotated_box(p1, p2, current_ratio)
    rect = cv2.boundingRect(full_poly_points)
    x, y, w, h = rect
    crop = frame[y:y+h, x:x+w]
    
    # 2. Re-calculate the HALF polygon for the Mask (Filtering) at full resolution
    c1, c2, c3, c4 = full_poly_points
    avg_y_12 = (c1[1] + c2[1]) / 2
    avg_y_34 = (c3[1] + c4[1]) / 2
    
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)

    if avg_y_12 < avg_y_34:
        half_poly = np.array([c1, c2, p2_arr, p1_arr], dtype=np.int32)
    else:
        half_poly = np.array([c4, c3, p2_arr, p1_arr], dtype=np.int32)
    
    # 3. Create Mask using ONLY the Top Half
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    relative_points = half_poly - [x, y]
    cv2.fillPoly(mask, [relative_points], 255)
    
    return rect, crop, mask, relative_points

# ==========================================
#      PART 2: SYSTEM CLASSES (THREADED)
# ==========================================
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
        self.thread = threading.Thread(target=self._reader, daemon=True)
    def start(self): self.thread.start(); return self
    def _reader(self):
        while not self.stopped:
            if self.seek_requested: time.sleep(0.01); continue
            if not self.q.full():
                ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                ret, frame = self.cap.read()
                if not ret: self.stopped = True; break
                self.q.put((frame, ts))
            else: time.sleep(0.001)
    def read(self):
        if self.q.empty(): return False, None, 0
        return True, *self.q.get()
    def seek(self, sec):
        self.seek_requested = True
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.cap.get(cv2.CAP_PROP_POS_MSEC) + sec*1000)
        with self.q.mutex: self.q.queue.clear()
        self.seek_requested = False
    def release(self): self.stopped = True; self.thread.join(); self.cap.release()

class AsyncVisualizer:
    def __init__(self, display_width):
        self.q = queue.Queue(maxsize=2)
        self.display_width = display_width
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def update(self, frame, fps_text, time_text):
        if not self.q.full():
            self.q.put((frame, fps_text, time_text))

    def _worker(self):
        cv2.namedWindow('Main View')
        while not self.stopped:
            try:
                frame, fps_text, time_text = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
            cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, time_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            h, w = frame.shape[:2]
            d_h = int(self.display_width * (h / w))
            display_frame = cv2.resize(frame, (self.display_width, d_h), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow('Main View', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
            
            self.q.task_done()
        cv2.destroyAllWindows()

def detect_sparks_yolo(frame, model, conf_thresh):
    spark_boxes = []
    results = model(frame, imgsz=MODEL_IMGSZ, conf=conf_thresh, verbose=False, half=True)
    for box in results[0].boxes:
        xywh = box.xywh[0].cpu().numpy()
        w, h = int(xywh[2]), int(xywh[3])
        if w < MIN_SPARK_WIDTH or h < MIN_SPARK_HEIGHT: continue 
        if w > MAX_SPARK_WIDTH or h > MAX_SPARK_HEIGHT: continue 
        x, y = int(xywh[0] - w/2), int(xywh[1] - h/2)
        spark_boxes.append((max(0, x), max(0, y), w, h))
    return spark_boxes

# ==========================================
#      PART 3: MAIN LOOP
# ==========================================
def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    saver = AsyncImageSaver()
    print(f"Loading Engine: {YOLO_MODEL_PATH}")
    spark_model = YOLO(YOLO_MODEL_PATH, task='detect')
    
    cap = FastVideoCapture(VIDEO_SOURCE).start()
    time.sleep(1.0)
    ret, first_frame, _ = cap.read()
    if not ret: return
    video_h, video_w = first_frame.shape[:2]

    # --- 1. SETUP ---
    select_scale = 1280 / video_w if video_w > 1280 else 1.0
    rect, template_raw_crop, template_mask_binary, rel_poly_points = get_interactive_roi(first_frame, scale=select_scale)
    
    x, y, w, h = rect
    orig_w, orig_h = w, h
    
    # --- SAVE REFERENCE TEMPLATE (ONCE) ---
    # This is the "Raw Template" used for tracking
    ref_template_path = f"{OUTPUT_FOLDER}/MASTER_TEMPLATE_{w}x{h}.jpg"
    cv2.imwrite(ref_template_path, template_raw_crop)
    print(f"\n[INFO] Saved Master Template to: {ref_template_path}")
    
    template_gray = cv2.cvtColor(template_raw_crop, cv2.COLOR_BGR2GRAY)
    
    small_w = int(orig_w * MATCHING_DOWNSCALE)
    small_h = int(orig_h * MATCHING_DOWNSCALE)
    
    current_downscale = MATCHING_DOWNSCALE
    if small_w < 25 or small_h < 25:
        print("⚠️ Warning: ROI too small. Increasing tracking quality automatically.")
        current_downscale = 0.5 
        small_w = int(orig_w * current_downscale)
        small_h = int(orig_h * current_downscale)
    
    upscale_factor = 1.0 / current_downscale
    small_template = cv2.resize(template_gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    viz = AsyncVisualizer(DISPLAY_WIDTH)

    print(f"Mode: ANNOTATED + RAW DETECTION")
    
    # Warmup
    dummy = np.zeros((MODEL_IMGSZ, MODEL_IMGSZ, 3), dtype='uint8')
    for _ in range(3): spark_model(dummy, imgsz=MODEL_IMGSZ, verbose=False, half=True)

    frames_processed_count = 0
    last_fps_time = time.time()
    display_fps = 0
    last_log_time = 0 
    
    panto_x, panto_y = x, y
    tracking_counter = 0
    
    last_loc_small = (int(x * current_downscale), int(y * current_downscale))

    while True:
        if viz.stopped: break

        ret, frame, current_ts_ms = cap.read()
        if not ret: 
            if cap.stopped: break
            time.sleep(0.005); continue

        frames_processed_count += 1
        tracking_counter += 1

        # --- 2. TRACKING ---
        should_track = (tracking_counter >= TRACKING_SKIP_FRAMES)
        
        if should_track:
            small_frame = cv2.resize(frame, (0, 0), fx=current_downscale, fy=current_downscale, interpolation=cv2.INTER_NEAREST)
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            if last_loc_small is None:
                search_crop = gray_small
                off_x, off_y = 0, 0
            else:
                lx, ly = last_loc_small
                lx, ly = int(lx), int(ly)
                margin = int(SEARCH_WINDOW_MARGIN * current_downscale)
                y1 = max(0, ly - margin); y2 = min(gray_small.shape[0], ly + small_h + margin)
                x1 = max(0, lx - margin); x2 = min(gray_small.shape[1], lx + small_w + margin)
                
                if x2 <= x1 or y2 <= y1: 
                    search_crop = gray_small
                    off_x, off_y = 0, 0
                else:
                    search_crop = gray_small[y1:y2, x1:x2]
                    off_x, off_y = x1, y1

            res = cv2.matchTemplate(search_crop, small_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > TRACKING_CONFIDENCE_THRESHOLD:
                best_loc_small = (max_loc[0] + off_x, max_loc[1] + off_y)
                last_loc_small = best_loc_small
                
                panto_x = int(best_loc_small[0] * upscale_factor)
                panto_y = int(best_loc_small[1] * upscale_factor)
            
            tracking_counter = 0 

        # --- 3. DETECTION ---
        roi_x1 = max(0, panto_x - ROI_PADDING)
        roi_y1 = max(0, panto_y - ROI_PADDING)
        roi_x2 = min(video_w, panto_x + orig_w + ROI_PADDING)
        roi_y2 = min(video_h, panto_y + orig_h + ROI_PADDING)
        
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            raw_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            if raw_roi.size > 0:
                roi_green = raw_roi[:, :, 1]
                _, thresh = cv2.threshold(roi_green, AI_TRIGGER_THRESHOLD, 255, cv2.THRESH_BINARY)
                
                if cv2.countNonZero(thresh) > 0:
                    filter_mask = np.zeros(raw_roi.shape[:2], dtype=np.uint8)
                    
                    # Calculate current mask position relative to ROI
                    current_poly_offset = rel_poly_points + [panto_x - roi_x1, panto_y - roi_y1]
                    cv2.fillPoly(filter_mask, [current_poly_offset], 255)
                    filter_mask = cv2.dilate(filter_mask, np.ones((MASK_EXPANSION, MASK_EXPANSION), np.uint8), iterations=1)
                    
                    spark_boxes = detect_sparks_yolo(raw_roi, spark_model, YOLO_CONFIDENCE_THRESHOLD)
                    
                    for (sx, sy, sw, sh) in spark_boxes:
                        center_x, center_y = sx + sw // 2, sy + sh // 2
                        
                        # --- STRICT FILTERING ---
                        # Only allow if the center is inside the "Top Half" mask
                        if center_y < filter_mask.shape[0] and center_x < filter_mask.shape[1]:
                            if filter_mask[center_y, center_x] == 0: continue
                        else:
                            continue

                        spark_crop_data = raw_roi[sy:sy+sh, sx:sx+sw]
                        if spark_crop_data.size == 0: continue
                        
                        peak_brightness = np.max(spark_crop_data)
                        if peak_brightness < SPARK_BRIGHTNESS_THRESHOLD: continue 

                        # --- SAVE ANNOTATED & RAW IMAGES ---
                        if (time.time() - last_log_time) > 1.0: 
                            time_str = format_timestamp(current_ts_ms)
                            
                            # 1. Annotated Image
                            annotated_img = raw_roi.copy()
                            cv2.polylines(annotated_img, [current_poly_offset.astype(np.int32)], True, (0, 255, 0), 2)
                            cv2.rectangle(annotated_img, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                            
                            # Size Categorization (Using Diagonal)
                            spark_diag = np.sqrt(sw**2 + sh**2)
                            
                            if spark_diag <= 20:
                                size_category = "Low"
                            elif spark_diag <= 40:
                                size_category = "Medium"
                            else:
                                size_category = "High"

                            info_text = f"Val:{peak_brightness} Size:{size_category}"
                            cv2.putText(annotated_img, info_text, (sx, sy - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                            print(f"\n🔥 SPARK DETECTED | Time: {time_str} | Brightness: {peak_brightness}/255 | Size: {size_category} (Diag:{int(spark_diag)}px)")
                            print(f"   Poly Points (ROI Relative): {current_poly_offset.tolist()}")
                            
                            saver.save(f"{OUTPUT_FOLDER}/spark_{time_str}_annotated.jpg", annotated_img)
                            
                            # 2. Raw Image (Clean ROI) - Saved alongside annotated version
                            saver.save(f"{OUTPUT_FOLDER}/spark_{time_str}_raw.jpg", raw_roi)
                            
                            last_log_time = time.time()

        # --- 4. VISUALIZER UPDATE (Clean) ---
        curr_time = time.time()
        if curr_time - last_fps_time >= 1.0:
            display_fps = frames_processed_count
            frames_processed_count = 0
            last_fps_time = curr_time

        fps_str = f"FPS: {display_fps}"
        time_str = f"Time: {format_timestamp(current_ts_ms).replace('_', ':')}"
        
        viz.update(frame, fps_text=fps_str, time_text=time_str)

    cap.release()
    viz.stopped = True

if __name__ == "__main__":
    main()
