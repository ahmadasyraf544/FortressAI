import os
import cv2
import glob
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ==============================================================================
# 1. CLASS ONTOLOGY SPECIFICATION (13 CLASSES FOR SEDIC 2026)
# ==============================================================================
CLASSES = [
    "container_ship",          # Class ID 0
    "tanker",                  # Class ID 1
    "cargo_ship",              # Class ID 2
    "passenger_ferry",         # Class ID 3
    "yacht",                   # Class ID 4
    "speedboat",               # Class ID 5
    "fishing_boat",            # Class ID 6
    "military_vessel_local",   # Class ID 7
    "military_vessel_foreign", # Class ID 8
    "military_vessel_general", # Class ID 9
    "military_helicopter",     # Class ID 10
    "military_airfighter",     # Class ID 11
    "military_submarine"       # Class ID 12
]

PROMPT_TO_CLASS_ID = {
    "container ship": 0,
    "oil tanker": 1,
    "cargo ship": 2,
    "passenger ferry": 3,
    "yacht": 4,
    "speedboat": 5,
    "fishing boat": 6,
    "local military vessel": 7,
    "foreign military vessel": 8,
    "military warship": 9,
    "military helicopter": 10,
    "attack helicopter": 10,
    "military jet": 11,
    "fighter jet": 11,
    "military aircraft": 11,
    "military submarine": 12,
    "submarine": 12
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_VIDEO_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DATASET_DIR, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DATASET_DIR, "labels")

# Object detection confidence threshold (0.25 = 25%)
CONFIDENCE_THRESHOLD = 0.25

# Frame step interval (Set 5 or 10 to avoid saving redundant frames)
FRAME_INTERVAL = 5

# Check GPU (CUDA) availability
if torch.cuda.is_available():
    DEVICE = 0  # Use GPU 0 (NVIDIA CUDA)
    DEVICE_NAME = torch.cuda.get_device_name(0)
    print(f"[HARDWARE] GPU Detected! Using GPU: {DEVICE_NAME}")
else:
    DEVICE = 'cpu'
    DEVICE_NAME = 'CPU'
    print("[HARDWARE WARNING] No CUDA GPU detected. Script will run on CPU.")

def setup_environment():
    """Ensure output directories exist and generate classes.txt and data.yaml."""
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    
    classes_file_dataset = os.path.join(OUTPUT_DATASET_DIR, "classes.txt")
    classes_file_labels = os.path.join(OUTPUT_LABELS_DIR, "classes.txt")
    classes_file_root = os.path.join(BASE_DIR, "classes.txt")

    with open(classes_file_dataset, "w", encoding="utf-8") as f:
        for cls_name in CLASSES:
            f.write(f"{cls_name}\n")
            
    with open(classes_file_labels, "w", encoding="utf-8") as f:
        for cls_name in CLASSES:
            f.write(f"{cls_name}\n")

    with open(classes_file_root, "w", encoding="utf-8") as f:
        for cls_name in CLASSES:
            f.write(f"{cls_name}\n")

    # Generate data.yaml for YOLO training
    yaml_path = os.path.join(OUTPUT_DATASET_DIR, "data.yaml")
    yaml_content = f"""# SEDIC 2026 Dataset Configuration for YOLO Training
path: {OUTPUT_DATASET_DIR}
train: images
val: images

names:
  0: container_ship
  1: tanker
  2: cargo_ship
  3: passenger_ferry
  4: yacht
  5: speedboat
  6: fishing_boat
  7: military_vessel_local
  8: military_vessel_foreign
  9: military_vessel_general
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[INFO] Dataset structure initialized at: {OUTPUT_DATASET_DIR}")

def process_videos():
    """
    Extract video frames, run YOLO object detection on GPU/CPU,
    filter out empty frames, and save image (.jpg) to images/ & label (.txt) to labels/.
    Includes auto-skip / resume capability for already processed frames.
    """
    setup_environment()
    
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(INPUT_VIDEO_DIR, ext)))
        video_files.extend(glob.glob(os.path.join(INPUT_VIDEO_DIR, ext.upper())))
        
    if not video_files:
        print(f"[WARNING] No video files found in directory: {INPUT_VIDEO_DIR}")
        return

    print(f"[INFO] Found {len(video_files)} video files. Starting processing...")

    model_name = "yolov8s-worldv2.pt"
    print(f"[INFO] Loading detection model: {model_name} onto {DEVICE_NAME}...")
    
    try:
        model = YOLO(model_name)
        if "world" in model_name:
            model.set_classes(list(PROMPT_TO_CLASS_ID.keys()))
            print("[INFO] Custom YOLO-World classes configured successfully.")
    except Exception as e:
        print(f"[WARNING] Failed to load {model_name}, falling back to standard yolov8n.pt. Error: {e}")
        model = YOLO("yolov8n.pt")

    total_frames_processed = 0
    total_saved_frames = 0
    total_skipped_frames = 0
    total_discarded_frames = 0
    total_bboxes_created = 0

    for video_path in video_files:
        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Unable to open video file: {video_path}")
            continue

        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n[PROCESSING VIDEO] {Path(video_path).name} (Total Frames: {total_vid_frames})")

        frame_count = 0
        
        with tqdm(total=total_vid_frames, desc=f"Processing {video_name}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                pbar.update(1)

                if frame_count % FRAME_INTERVAL != 0:
                    continue

                # Resume feature: Check if image and label files already exist in subfolders
                base_filename = f"{video_name}_frame_{frame_count:06d}"
                img_output_path = os.path.join(OUTPUT_IMAGES_DIR, f"{base_filename}.jpg")
                txt_output_path = os.path.join(OUTPUT_LABELS_DIR, f"{base_filename}.txt")

                if os.path.exists(img_output_path) and os.path.exists(txt_output_path):
                    total_skipped_frames += 1
                    total_saved_frames += 1
                    continue

                total_frames_processed += 1
                img_h, img_w, _ = frame.shape

                # Run YOLO inference
                results = model.predict(
                    source=frame, 
                    conf=CONFIDENCE_THRESHOLD, 
                    device=DEVICE, 
                    verbose=False
                )[0]

                yolo_annotations = []

                if results.boxes is not None and len(results.boxes) > 0:
                    for box in results.boxes:
                        cls_id_pred = int(box.cls[0].item())
                        
                        if "world" in model_name and hasattr(model, 'custom_classes'):
                            prompt_keys = list(PROMPT_TO_CLASS_ID.keys())
                            if cls_id_pred < len(prompt_keys):
                                prompt_key = prompt_keys[cls_id_pred]
                                sedic_cls_id = PROMPT_TO_CLASS_ID.get(prompt_key, 6)
                            else:
                                sedic_cls_id = 6
                        else:
                            if cls_id_pred == 8:
                                sedic_cls_id = 6
                            elif cls_id_pred < len(CLASSES):
                                sedic_cls_id = cls_id_pred
                            else:
                                continue

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        w_px = x2 - x1
                        h_px = y2 - y1
                        xc = (x1 + w_px / 2.0) / img_w
                        yc = (y1 + h_px / 2.0) / img_h
                        nw = w_px / img_w
                        nh = h_px / img_h

                        xc, yc = max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc))
                        nw, nh = max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))

                        yolo_annotations.append(f"{sedic_cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

                # Frame Filtering & Label Saving
                if len(yolo_annotations) > 0:
                    cv2.imwrite(img_output_path, frame)

                    with open(txt_output_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(yolo_annotations) + "\n")

                    total_saved_frames += 1
                    total_bboxes_created += len(yolo_annotations)
                else:
                    total_discarded_frames += 1

        cap.release()

    print("\n=======================================================")
    print("FRAME EXTRACTION & AUTO-LABELING COMPLETED!")
    print("=======================================================")
    print(f"Device Used                 : {DEVICE_NAME}")
    print(f"Total Video Files Processed : {len(video_files)}")
    print(f"New Frames Processed        : {total_frames_processed}")
    print(f"Skipped Frames (Existing)   : {total_skipped_frames}")
    print(f"Total Saved Frames (images/): {total_saved_frames}")
    print(f"Discarded Frames (Empty)    : {total_discarded_frames}")
    print(f"Total Bounding Boxes        : {total_bboxes_created}")
    print(f"Images Output Directory     : {os.path.abspath(OUTPUT_IMAGES_DIR)}")
    print(f"Labels Output Directory     : {os.path.abspath(OUTPUT_LABELS_DIR)}")
    print("=======================================================")

if __name__ == "__main__":
    process_videos()