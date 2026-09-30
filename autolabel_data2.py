import os
import glob
import re
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# SEDIC 2026 - AUTOLABEL SCRIPT FOR VIDEOS_DATA2 (FOLDER-BASED CLASS ASSIGNMENT)
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
    "military_submarine",      # 12
    "tug_boat"                 # 13
]

# Mapping keyword in folder name to exact SEDIC Class ID
FOLDER_TO_CLASS_ID = {
    "foreign military vessel": 8,
    "military_vessel_foreign": 8,
    "cargo ship": 2,
    "cargo_ship": 2,
    "container ship": 0,
    "container_ship": 0,
    "fishing boat": 6,
    "fishing_boat": 6,
    "passenger ferry": 3,
    "passenger_ferry": 3,
    "speed boat": 5,
    "speedboat": 5,
    "tanker": 1,
    "oil tanker": 1,
    "oil_tanker": 1,
    "yacht": 4,
    "military_vessel_local": 7,
    "military_vessel_general": 9,
    "military_helicopter": 10,
    "helicopter": 10,
    "military_airfighter": 11,
    "airfighter": 11,
    "fighter": 11,
    "military_submarine": 12,
    "submarine": 12,
    "tug boat": 13,
    "tug_boat": 13,
    "tugboat": 13
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DATA2_DIR = os.path.join(BASE_DIR, "Videos_Dataset", "Videos_Data2")
CONFIDENCE_THRESHOLD = 0.18

# Hardware configuration
if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = torch.cuda.get_device_name(0)
    print(f"[HARDWARE] GPU Detected: {DEVICE_NAME} (Using CUDA:0)")
else:
    DEVICE = 'cpu'
    DEVICE_NAME = 'CPU'
    print("[HARDWARE WARNING] No CUDA GPU detected. Running on CPU.")

def get_class_id_from_folder_name(folder_name):
    clean_name = re.sub(r'^\d+\.\s*', '', folder_name).lower().strip()
    for key, cid in FOLDER_TO_CLASS_ID.items():
        if key in clean_name:
            return cid
    return None

def autolabel_videos_data2():
    if not os.path.exists(VIDEOS_DATA2_DIR):
        print(f"[ERROR] Videos_Data2 directory not found: {VIDEOS_DATA2_DIR}")
        return

    # Load YOLO detection model for vessel bounding box localization
    model_name = "yolov8s-worldv2.pt" if os.path.exists("yolov8s-worldv2.pt") else "yolov8s.pt"
    print(f"[INFO] Loading detection model: {model_name} onto {DEVICE_NAME}...")
    model = YOLO(model_name)
    
    subfolders = sorted([f for f in os.listdir(VIDEOS_DATA2_DIR) if os.path.isdir(os.path.join(VIDEOS_DATA2_DIR, f))])
    print(f"[INFO] Found {len(subfolders)} class folders in Videos_Data2 to auto-label.\n")

    total_images_processed = 0
    total_labels_written = 0
    total_bboxes = 0
    stats_per_folder = []

    for idx, folder in enumerate(subfolders, 1):
        folder_path = os.path.join(VIDEOS_DATA2_DIR, folder)
        imgs_dir = os.path.join(folder_path, "images")
        lbls_dir = os.path.join(folder_path, "labels")

        if not os.path.exists(imgs_dir):
            print(f"[{idx}/{len(subfolders)}] [SKIP] No images folder in: {folder}")
            continue

        os.makedirs(lbls_dir, exist_ok=True)
        class_id = get_class_id_from_folder_name(folder)
        class_name = CLASSES[class_id] if class_id is not None else "Unknown"

        # Update classes.txt in folder and labels/
        classes_txt_content = "\n".join(CLASSES) + "\n"
        with open(os.path.join(folder_path, "classes.txt"), "w", encoding="utf-8") as f:
            f.write(classes_txt_content)
        with open(os.path.join(lbls_dir, "classes.txt"), "w", encoding="utf-8") as f:
            f.write(classes_txt_content)

        # Update data.yaml in folder
        abs_fpath = os.path.abspath(folder_path).replace('\\', '/')
        yaml_lines = [
            f"# SEDIC 2026 Dataset - Videos_Data2: {folder}",
            f"path: {abs_fpath}",
            f"train: images",
            f"val: images",
            "",
            "names:"
        ]
        for cid, cname in enumerate(CLASSES):
            yaml_lines.append(f"  {cid}: {cname}")

        with open(os.path.join(folder_path, "data.yaml"), "w", encoding="utf-8") as f:
            f.write("\n".join(yaml_lines) + "\n")

        image_files = sorted(glob.glob(os.path.join(imgs_dir, "*.jpg")) + glob.glob(os.path.join(imgs_dir, "*.png")))
        print(f"[{idx}/{len(subfolders)}] Auto-labeling: {folder}")
        print(f"     Class ID: {class_id} ({class_name}) | Total Images: {len(image_files)}")

        f_labeled = 0
        f_bboxes = 0
        
        # Batch inference in chunks of 16 for maximum GPU throughput
        batch_size = 16
        for i in tqdm(range(0, len(image_files), batch_size), desc=f"  {folder[:20]}", unit="batch"):
            batch_paths = image_files[i:i + batch_size]
            results = model.predict(
                source=batch_paths,
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

            for img_path, res in zip(batch_paths, results):
                stem = Path(img_path).stem
                lbl_path = os.path.join(lbls_dir, f"{stem}.txt")
                
                img_h, img_w = res.orig_shape
                annotations = []

                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w_px = x2 - x1
                        h_px = y2 - y1

                        # Filter out tiny artifacts
                        if w_px < 15 or h_px < 15:
                            continue

                        xc = (x1 + w_px / 2.0) / img_w
                        yc = (y1 + h_px / 2.0) / img_h
                        nw = w_px / img_w
                        nh = h_px / img_h

                        xc, yc = max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc))
                        nw, nh = max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))

                        # If folder has defined target class, use it
                        cid = class_id if class_id is not None else int(box.cls[0].item())
                        annotations.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

                # If no bounding box detected, default a central bounding box heuristic
                if len(annotations) == 0 and class_id is not None:
                    annotations.append(f"{class_id} 0.500000 0.550000 0.650000 0.450000")

                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(annotations) + "\n")

                f_labeled += 1
                f_bboxes += len(annotations)

        total_images_processed += len(image_files)
        total_labels_written += f_labeled
        total_bboxes += f_bboxes

        stats_per_folder.append({
            "folder": folder,
            "class_id": class_id,
            "class_name": class_name,
            "images": len(image_files),
            "bboxes": f_bboxes
        })
        print(f"     -> Labeled: {f_labeled} files | Total BBoxes: {f_bboxes}\n")

    # Update collection data.yaml in Videos_Data2
    abs_v2 = os.path.abspath(VIDEOS_DATA2_DIR).replace('\\', '/')
    yaml_lines_master = [
        f"# SEDIC 2026 Dataset - Videos_Data2 Complete Collection",
        f"path: {abs_v2}",
        f"train: train.txt",
        f"val: val.txt",
        "",
        "names:"
    ]
    for cid, cname in enumerate(CLASSES):
        yaml_lines_master.append(f"  {cid}: {cname}")

    with open(os.path.join(VIDEOS_DATA2_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines_master) + "\n")

    # Update Master train.txt and val.txt in Videos_Dataset
    master_dir = os.path.join(BASE_DIR, "Videos_Dataset")
    all_master_imgs = sorted(glob.glob(os.path.join(master_dir, "**", "*.jpg"), recursive=True))
    import random
    random.seed(42)
    random.shuffle(all_master_imgs)
    split_idx = int(len(all_master_imgs) * 0.8)

    with open(os.path.join(master_dir, "train.txt"), "w", encoding="utf-8") as f:
        for p in all_master_imgs[:split_idx]:
            f.write(f"{os.path.abspath(p)}\n")

    with open(os.path.join(master_dir, "val.txt"), "w", encoding="utf-8") as f:
        for p in all_master_imgs[split_idx:]:
            f.write(f"{os.path.abspath(p)}\n")

    print("="*80)
    print("AUTOLABELING FOR VIDEOS_DATA2 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Total Images Auto-Labeled : {total_images_processed:,}")
    print(f"Total Bounding Boxes Created: {total_bboxes:,}")
    print(f"Master Dataset Images Count: {len(all_master_imgs):,}")
    print("="*80)
    for s in stats_per_folder:
        print(f" - {s['folder']:30s} -> ID {s['class_id']}: {s['class_name']:20s} ({s['images']:,} imgs, {s['bboxes']:,} bboxes)")
    print("="*80)

def autolabel_tugboat_only():
    """Autolabels exclusively the 09. tug boat folder."""
    tugboat_folder = os.path.join(VIDEOS_DATA2_DIR, "09. tug boat")
    if not os.path.exists(tugboat_folder):
        print(f"[ERROR] 09. tug boat folder not found: {tugboat_folder}")
        return

    imgs_dir = os.path.join(tugboat_folder, "images")
    lbls_dir = os.path.join(tugboat_folder, "labels")
    os.makedirs(lbls_dir, exist_ok=True)

    # Update classes.txt
    classes_txt_content = "\n".join(CLASSES) + "\n"
    with open(os.path.join(tugboat_folder, "classes.txt"), "w", encoding="utf-8") as f:
        f.write(classes_txt_content)
    with open(os.path.join(lbls_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write(classes_txt_content)

    # Load model
    model_name = "yolov8s-worldv2.pt" if os.path.exists("yolov8s-worldv2.pt") else "yolov8s.pt"
    print(f"[INFO] Loading detection model: {model_name} onto {DEVICE_NAME}...")
    model = YOLO(model_name)
    if "world" in model_name:
        try:
            model.set_classes(["tugboat", "tug boat", "harbor tug", "boat", "vessel", "ship"])
        except Exception:
            pass

    image_files = sorted(glob.glob(os.path.join(imgs_dir, "*.jpg")) + glob.glob(os.path.join(imgs_dir, "*.png")))
    print(f"[INFO] Auto-labeling 09. tug boat exclusively (Target Class ID: 13 - tug_boat)")
    print(f"       Total Images: {len(image_files):,}")

    batch_size = 16
    f_labeled = 0
    f_bboxes = 0

    for i in tqdm(range(0, len(image_files), batch_size), desc="Tug Boat Labeling", unit="batch"):
        batch_paths = image_files[i:i + batch_size]
        results = model.predict(
            source=batch_paths,
            conf=CONFIDENCE_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        for img_path, res in zip(batch_paths, results):
            stem = Path(img_path).stem
            lbl_path = os.path.join(lbls_dir, f"{stem}.txt")
            
            img_h, img_w = res.orig_shape
            annotations = []

            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w_px = x2 - x1
                    h_px = y2 - y1

                    if w_px < 15 or h_px < 15:
                        continue

                    xc = (x1 + w_px / 2.0) / img_w
                    yc = (y1 + h_px / 2.0) / img_h
                    nw = w_px / img_w
                    nh = h_px / img_h

                    xc, yc = max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc))
                    nw, nh = max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))

                    annotations.append(f"13 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            if len(annotations) == 0:
                # Default clean central box fallback
                annotations.append(f"13 0.500000 0.550000 0.650000 0.450000")

            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(annotations) + "\n")

            f_labeled += 1
            f_bboxes += len(annotations)

    print("\n" + "=" * 70)
    print("✅ TUG BOAT AUTO-LABELING COMPLETED!")
    print(f" - Labeled Images: {f_labeled:,}")
    print(f" - Total BBoxes  : {f_bboxes:,} (Class ID 13: tug_boat)")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-label Videos_Data2")
    parser.add_argument("--tugboat-only", action="store_true", help="Auto-label tug boat folder only")
    args = parser.parse_args()

    if args.tugboat_only:
        autolabel_tugboat_only()
    else:
        autolabel_videos_data2()

