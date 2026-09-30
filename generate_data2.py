import os
import cv2
import glob
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# SEDIC 2026 - DATASET GENERATION FOR VIDEOS_DATA2 (3 FRAMES PER SECOND)
# ==============================================================================

CLASSES = [
    "container_ship",          # 0
    "tanker",                  # 1
    "cargo_ship",              # 2
    "passenger_ferry",         # 3
    "yacht",                   # 4
    "speedboat",               # 5
    "fishing_boat",            # 6
    "military_vessel_local",   # 7
    "military_vessel_foreign", # 8
    "military_vessel_general", # 9
    "military_helicopter",     # 10
    "military_airfighter",     # 11
    "military_submarine"       # 12
]

# Map video stem names directly to their target ground-truth class ID
VIDEO_TO_CLASS_ID = {
    "foreign military vessel": 8,
    "cargo ship": 2,
    "container ship": 0,
    "fishing boat": 6,
    "passenger ferry": 3,
    "speed boat": 5,
    "tanker": 1,
    "yacht": 4,
    "military helicopter": 10,
    "military airfighter": 11,
    "military submarine": 12,
    "submarine": 12
}

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
INPUT_VIDEO_DIR = os.path.join(BASE_DIR, "videos", "Videos_Data2")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "Videos_Dataset", "Videos_Data2")

TARGET_FPS = 3.0
CONFIDENCE_THRESHOLD = 0.20

# Hardware configuration
if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = torch.cuda.get_device_name(0)
    print(f"[HARDWARE] GPU Detected: {DEVICE_NAME} (Using CUDA:0)")
else:
    DEVICE = 'cpu'
    DEVICE_NAME = 'CPU'
    print("[HARDWARE WARNING] No CUDA GPU detected. Running on CPU.")

def get_target_class_for_video(video_stem):
    v_lower = video_stem.lower().strip()
    for key, cid in VIDEO_TO_CLASS_ID.items():
        if key in v_lower:
            return cid
    return None

def process_videos_data2():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    video_files = sorted(glob.glob(os.path.join(INPUT_VIDEO_DIR, "*.mp4")))
    if not video_files:
        print(f"[ERROR] No video files found in: {INPUT_VIDEO_DIR}")
        return

    print(f"[INFO] Found {len(video_files)} videos to process at {TARGET_FPS} FPS.")
    
    # Load YOLO Model
    model_name = "yolov8s-worldv2.pt" if os.path.exists("yolov8s-worldv2.pt") else "yolov8n.pt"
    print(f"[INFO] Loading detection model: {model_name}...")
    model = YOLO(model_name)
    if "world" in model_name:
        try:
            model.set_classes(list(PROMPT_TO_CLASS_ID.keys()))
            print("[INFO] YOLO-World classes configured.")
        except Exception as e:
            print(f"[WARNING] Could not set custom classes: {e}")

    # Generate master classes.txt in Videos_Data2/
    master_classes_file = os.path.join(OUTPUT_BASE_DIR, "classes.txt")
    with open(master_classes_file, "w", encoding="utf-8") as f:
        for c in CLASSES:
            f.write(f"{c}\n")

    video_stats = []
    total_saved = 0
    total_discarded = 0
    total_skipped = 0

    for idx, vpath in enumerate(video_files, 1):
        vstem = Path(vpath).stem
        default_cid = get_target_class_for_video(vstem)
        
        # Create dedicated subfolder for this video
        folder_name = f"{idx:02d}. {vstem}"
        video_out_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
        imgs_dir = os.path.join(video_out_dir, "images")
        lbls_dir = os.path.join(video_out_dir, "labels")
        os.makedirs(imgs_dir, exist_ok=True)
        os.makedirs(lbls_dir, exist_ok=True)
        
        # Write classes.txt inside video subfolder and labels/
        classes_txt_content = "\n".join(CLASSES) + "\n"
        with open(os.path.join(video_out_dir, "classes.txt"), "w", encoding="utf-8") as f:
            f.write(classes_txt_content)
        with open(os.path.join(lbls_dir, "classes.txt"), "w", encoding="utf-8") as f:
            f.write(classes_txt_content)

        # Write data.yaml inside video subfolder
        abs_vdir = os.path.abspath(video_out_dir).replace('\\', '/')
        v_yaml_file = os.path.join(video_out_dir, "data.yaml")
        yaml_content = f"""# SEDIC 2026 Dataset - Videos_Data2: {vstem}
path: {abs_vdir}
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
        with open(v_yaml_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {vpath}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps / TARGET_FPS))
        
        print(f"\n[{idx}/{len(video_files)}] Processing: {Path(vpath).name}")
        print(f"     FPS: {fps:.2f} | Extract Step: Every {step} frames (~{TARGET_FPS} fps) | Class ID: {default_cid} ({CLASSES[default_cid] if default_cid is not None else 'Generic'})")

        frame_count = 0
        v_saved = 0
        v_discarded = 0
        v_skipped = 0

        with tqdm(total=total_vid_frames, desc=f"{vstem[:25]}", unit="frames") as pbar:
            while cap.isOpened() and frame_count < total_vid_frames:
                # Fast forward (step - 1) frames without decoding pixels
                for _ in range(step - 1):
                    if not cap.grab():
                        break
                    frame_count += 1
                    pbar.update(1)

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                pbar.update(1)

                base_filename = f"{vstem}_frame_{frame_count:06d}"
                img_path = os.path.join(imgs_dir, f"{base_filename}.jpg")
                txt_path = os.path.join(lbls_dir, f"{base_filename}.txt")

                # Resume capability
                if os.path.exists(img_path) and os.path.exists(txt_path):
                    v_skipped += 1
                    v_saved += 1
                    continue

                img_h, img_w, _ = frame.shape

                # YOLO Inference
                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    device=DEVICE,
                    verbose=False
                )[0]

                annotations = []
                if results.boxes is not None and len(results.boxes) > 0:
                    for box in results.boxes:
                        cls_id_pred = int(box.cls[0].item())
                        
                        # Determine class ID
                        if default_cid is not None:
                            sedic_cls_id = default_cid
                        elif "world" in model_name and hasattr(model, 'custom_classes'):
                            prompt_keys = list(PROMPT_TO_CLASS_ID.keys())
                            if cls_id_pred < len(prompt_keys):
                                prompt_key = prompt_keys[cls_id_pred]
                                sedic_cls_id = PROMPT_TO_CLASS_ID.get(prompt_key, 6)
                            else:
                                sedic_cls_id = 6
                        else:
                            sedic_cls_id = cls_id_pred if cls_id_pred < len(CLASSES) else 6

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w_px = x2 - x1
                        h_px = y2 - y1
                        xc = (x1 + w_px / 2.0) / img_w
                        yc = (y1 + h_px / 2.0) / img_h
                        nw = w_px / img_w
                        nh = h_px / img_h

                        xc, yc = max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc))
                        nw, nh = max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))

                        annotations.append(f"{sedic_cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

                # If detected, save frame and label
                if len(annotations) > 0:
                    cv2.imwrite(img_path, frame)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(annotations) + "\n")
                    v_saved += 1
                else:
                    v_discarded += 1

        cap.release()
        total_saved += v_saved
        total_discarded += v_discarded
        total_skipped += v_skipped
        
        video_stats.append({
            "name": vstem,
            "saved": v_saved,
            "discarded": v_discarded
        })
        print(f"     -> Saved: {v_saved} frames | Discarded empty: {v_discarded}")

    # Generate collection data.yaml in Videos_Data2/
    abs_out_dir = os.path.abspath(OUTPUT_BASE_DIR).replace('\\', '/')
    collection_yaml = os.path.join(OUTPUT_BASE_DIR, "data.yaml")
    with open(collection_yaml, "w", encoding="utf-8") as f:
        f.write(f"""# SEDIC 2026 Dataset - Videos_Data2 Complete Collection
path: {abs_out_dir}
train: train.txt
val: val.txt

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
""")

    # Update Master train.txt and val.txt in Videos_Dataset/
    all_imgs = sorted(glob.glob(os.path.join(BASE_DIR, "Videos_Dataset", "**", "*.jpg"), recursive=True))
    import random
    random.seed(42)
    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * 0.8)
    
    with open(os.path.join(BASE_DIR, "Videos_Dataset", "train.txt"), "w", encoding="utf-8") as f:
        for p in all_imgs[:split_idx]:
            f.write(f"{os.path.abspath(p)}\n")
            
    with open(os.path.join(BASE_DIR, "Videos_Dataset", "val.txt"), "w", encoding="utf-8") as f:
        for p in all_imgs[split_idx:]:
            f.write(f"{os.path.abspath(p)}\n")

    print("\n" + "="*80)
    print("VIDEOS_DATA2 GENERATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Output Directory: {OUTPUT_BASE_DIR}")
    print(f"Total Saved Images & Labels: {total_saved:,}")
    print(f"Total Master Dataset Images: {len(all_imgs):,}")
    print("="*80)
    for s in video_stats:
        print(f" - {s['name']:30s} -> {s['saved']:,} frames saved")
    print("="*80)

if __name__ == "__main__":
    process_videos_data2()
