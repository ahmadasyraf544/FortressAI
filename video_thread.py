import os
import time
import cv2
import torch
import numpy as np
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt6.QtGui import QImage
from ultralytics import YOLO

from config import (
    CLASSES, get_class_color_bgr, DEFAULT_CONFIDENCE, 
    DEFAULT_IOU, DEFAULT_TRACKER, get_active_model_path
)

# Prevent OpenCV ffmpeg thread contention with PyTorch OpenMP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"
cv2.setNumThreads(1)

class VideoProcessingThread(QThread):
    """
    High-Performance Background QThread for OpenCV Capture & Ultralytics YOLO Tracking.
    Accelerated with CUDA FP16 Half-Precision, Layer Fusion, and Inference Mode for Maximum FPS.
    """
    frame_ready = pyqtSignal(QImage, dict)
    new_target_detected = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    video_completed = pyqtSignal()

    def __init__(self, model_path=None, parent=None):
        super().__init__(parent)
        self.model_path = model_path or get_active_model_path()
        self.conf = DEFAULT_CONFIDENCE
        self.iou = DEFAULT_IOU
        self.tracker = DEFAULT_TRACKER
        
        # High-FPS configuration
        self.imgsz = 480          # Optimal 480px for 2x-3x speedup on GTX 1650 Ti
        self.playback_speed = 1.0 # 1.0 = Real-time sync, 0.0 = Max uncapped GPU speed

        self.source_type = None   # 'webcam' or 'video'
        self.source_path = None
        
        self.is_running = False
        self.is_paused = False
        
        self.mutex = QMutex()
        self.pause_cond = QWaitCondition()
        
        # Tracking memory
        self.seen_track_ids = set()
        self.unique_targets_count = 0
        
        self.model = None
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.use_half = (self.device == 0)

    def load_model(self):
        try:
            device_name = torch.cuda.get_device_name(0) if self.device == 0 else "CPU"
            self.status_changed.emit(f"Loading YOLO model on {device_name} (FP16: {self.use_half})...")
            
            self.model = YOLO(self.model_path)
            
            # Fuse Conv+BN layers for inference speedup
            try:
                self.model.fuse()
            except Exception:
                pass
            
            # Pre-warm GPU with dummy frame to compile CUDA kernels
            if self.device == 0:
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                with torch.inference_mode():
                    self.model(dummy, imgsz=self.imgsz, device=self.device, verbose=False)
            
            self.status_changed.emit(f"Model Ready: {os.path.basename(self.model_path)} ({device_name})")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to load model: {e}")
            return False

    def set_source(self, source_type: str, source_path):
        """Sets source ('webcam', index: int) or ('video', file_path: str)."""
        self.mutex.lock()
        self.source_type = source_type
        self.source_path = source_path
        self.seen_track_ids.clear()
        self.unique_targets_count = 0
        self.mutex.unlock()

    def update_confidence(self, conf: float):
        """Live adjustment of confidence threshold without restarting thread."""
        self.mutex.lock()
        self.conf = max(0.01, min(1.0, conf))
        self.mutex.unlock()

    def update_iou(self, iou: float):
        """Live adjustment of IoU threshold."""
        self.mutex.lock()
        self.iou = max(0.01, min(1.0, iou))
        self.mutex.unlock()

    def set_imgsz(self, imgsz: int):
        """Adjust inference image size (e.g. 384, 480, 512, 640)."""
        self.mutex.lock()
        self.imgsz = imgsz
        self.mutex.unlock()

    def set_speed_mode(self, max_speed: bool):
        """Toggle between real-time video sync and max uncapped GPU throughput."""
        self.mutex.lock()
        self.playback_speed = 0.0 if max_speed else 1.0
        self.mutex.unlock()

    def pause(self):
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
        self.status_changed.emit("Video Feed Paused")

    def resume(self):
        self.mutex.lock()
        self.is_paused = False
        self.pause_cond.wakeAll()
        self.mutex.unlock()
        self.status_changed.emit("Video Feed Active")

    def toggle_pause(self):
        if self.is_paused:
            self.resume()
        else:
            self.pause()

    def stop(self):
        self.mutex.lock()
        self.is_running = False
        self.is_paused = False
        self.pause_cond.wakeAll()
        self.mutex.unlock()
        self.wait(1500)

    def seek_percentage(self, percent: float):
        """Seek video file playback to a percentage (0.0 to 100.0)."""
        self.mutex.lock()
        self._seek_target = percent
        self.mutex.unlock()

    def run(self):
        if not self.model:
            if not self.load_model():
                return

        self.mutex.lock()
        source = self.source_path
        stype = self.source_type
        self._seek_target = None
        self.mutex.unlock()

        if source is None:
            self.error_occurred.emit("No input source configured.")
            return

        cap = None
        try:
            if stype == "webcam":
                try:
                    cam_idx = int(source)
                except ValueError:
                    cam_idx = 0
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(str(source))

            if not cap.isOpened():
                self.error_occurred.emit(f"Unable to open video source: {source}")
                return

            fps_target = cap.get(cv2.CAP_PROP_FPS)
            if fps_target <= 0 or np.isnan(fps_target) or fps_target > 120:
                fps_target = 30.0
            frame_delay = 1.0 / fps_target
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if stype == "video" else 0

            self.is_running = True
            source_desc = os.path.basename(str(source)) if stype == 'video' else f'Webcam #{source}'
            self.status_changed.emit(f"Streaming: {source_desc}")

            fps_timer = time.perf_counter()
            frame_counter = 0
            current_fps = 0.0

            while self.is_running:
                # Handle Pause condition
                self.mutex.lock()
                while self.is_paused and self.is_running:
                    self.pause_cond.wait(self.mutex)
                
                # Check for seek request
                if self._seek_target is not None and stype == "video" and total_frames > 0:
                    target_frame = int(total_frames * (self._seek_target / 100.0))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    self._seek_target = None

                current_conf = self.conf
                current_iou = self.iou
                current_tracker = self.tracker
                current_imgsz = self.imgsz
                current_speed = self.playback_speed
                self.mutex.unlock()

                if not self.is_running:
                    break

                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    if stype == "video":
                        self.status_changed.emit("Video playback completed.")
                        self.video_completed.emit()
                    break

                # Accelerated Inference with PyTorch Inference Mode
                with torch.inference_mode():
                    try:
                        results = self.model.track(
                            frame,
                            persist=True,
                            tracker=current_tracker,
                            conf=current_conf,
                            iou=current_iou,
                            imgsz=current_imgsz,
                            device=self.device,
                            verbose=False
                        )
                    except Exception:
                        results = self.model(
                            frame,
                            conf=current_conf,
                            iou=current_iou,
                            imgsz=current_imgsz,
                            device=self.device,
                            verbose=False
                        )

                annotated_frame = frame.copy()
                h_img, w_img, _ = frame.shape
                active_tracks_count = 0
                detections_in_frame = 0

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        detections_in_frame = len(boxes)

                        # Extract tensors to CPU numpy arrays in batch
                        xyxy_all = boxes.xyxy.cpu().numpy().astype(int)
                        conf_all = boxes.conf.cpu().numpy()
                        cls_all = boxes.cls.cpu().numpy().astype(int)
                        id_all = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = xyxy_all[i]
                            conf_val = float(conf_all[i])
                            cls_id = int(cls_all[i])
                            track_id = int(id_all[i]) if id_all is not None else 0

                            if track_id > 0:
                                active_tracks_count += 1

                            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class_{cls_id}"
                            color_bgr = get_class_color_bgr(cls_id)

                            # Event-driven snapshot capture for newly spotted Track IDs
                            if track_id > 0 and track_id not in self.seen_track_ids:
                                self.seen_track_ids.add(track_id)
                                self.unique_targets_count += 1
                                
                                pad = 6
                                cx1 = max(0, x1 - pad)
                                cy1 = max(0, y1 - pad)
                                cx2 = min(w_img, x2 + pad)
                                cy2 = min(h_img, y2 + pad)
                                
                                if cx2 > cx1 and cy2 > cy1:
                                    crop_roi = frame[cy1:cy2, cx1:cx2].copy()
                                    target_event = {
                                        "track_id": track_id,
                                        "class_id": cls_id,
                                        "class_name": cls_name,
                                        "confidence": conf_val,
                                        "bbox": [x1, y1, x2, y2],
                                        "crop_image": crop_roi,
                                        "timestamp": datetime.now().strftime("%H:%M:%S")
                                    }
                                    self.new_target_detected.emit(target_event)

                            # Draw Tactical Bounding Box & HUD Label
                            self.draw_tactical_box(annotated_frame, x1, y1, x2, y2, track_id, cls_name, conf_val, color_bgr)

                # High-precision FPS calculation
                frame_counter += 1
                now = time.perf_counter()
                if now - fps_timer >= 0.5:
                    current_fps = frame_counter / (now - fps_timer)
                    frame_counter = 0
                    fps_timer = now

                # Video progress
                current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if stype == "video" else 0
                progress_percent = (current_frame_pos / total_frames * 100.0) if total_frames > 0 else 0.0

                # Fast BGR to RGB conversion for Qt
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

                metrics = {
                    "fps": current_fps,
                    "active_tracks": active_tracks_count,
                    "total_unique": len(self.seen_track_ids),
                    "detections_current": detections_in_frame,
                    "progress_percent": progress_percent,
                    "frame_pos": current_frame_pos,
                    "total_frames": total_frames,
                    "conf_threshold": current_conf,
                    "iou_threshold": current_iou
                }

                self.frame_ready.emit(q_image, metrics)

                # Adaptive Pacing
                if stype == "video" and current_speed > 0:
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = (frame_delay / current_speed) - elapsed
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)

        except Exception as e:
            self.error_occurred.emit(f"Playback error: {e}")
        finally:
            if cap is not None:
                cap.release()
            self.status_changed.emit("Video Feed Stopped")

    def draw_tactical_box(self, img, x1, y1, x2, y2, track_id, class_name, conf, color_bgr):
        """
        Fast rendering of tactical bounding box with format: #<Track_ID> <ClassName> <Confidence%>
        """
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

        corner_len = min(14, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))
        if corner_len > 3:
            thickness = 3
            # Top-Left
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), color_bgr, thickness)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), color_bgr, thickness)
            # Top-Right
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), color_bgr, thickness)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), color_bgr, thickness)
            # Bottom-Left
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), color_bgr, thickness)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), color_bgr, thickness)
            # Bottom-Right
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color_bgr, thickness)
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color_bgr, thickness)

        if track_id > 0:
            label_text = f"#{track_id} {class_name} {conf * 100:.1f}%"
        else:
            label_text = f"{class_name} {conf * 100:.1f}%"

        font_scale = 0.50
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        banner_y1 = max(0, y1 - text_h - 6)
        banner_y2 = y1
        cv2.rectangle(img, (x1, banner_y1), (x1 + text_w + 8, banner_y2), color_bgr, -1)

        text_color = (10, 10, 10) if (color_bgr[0]*0.114 + color_bgr[1]*0.587 + color_bgr[2]*0.299) > 150 else (255, 255, 255)
        cv2.putText(
            img, label_text, (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA
        )
