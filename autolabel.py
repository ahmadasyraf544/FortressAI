import os
import shutil
import glob
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ==============================================================================
# 1. CLASS ONTOLOGY SPECIFICATION (10 CLASSES FOR SEDIC 2026)
# ==============================================================================
CLASSES = [
    "container_ship",          # Class ID 0
    "tanker",                  # Class ID 1
    "cargo_ship",               # Class ID 2
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
    "military fighter jet": 11,
    "military submarine": 12
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default paths
DEFAULT_INPUT_IMAGES_DIR = r"D:\PLUS\SEDIC2026\dataset\images\train"
DEFAULT_OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CONFIDENCE_THRESHOLD = 0.25

# Hardware acceleration setup
if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = torch.cuda.get_device_name(0)
    print(f"[HARDWARE] GPU Detected! Using GPU: {DEVICE_NAME}")
else:
    DEVICE = 'cpu'
    DEVICE_NAME = 'CPU'
    print("[HARDWARE WARNING] No CUDA GPU detected. Running on CPU.")

def process_image_folder(
    input_dir=DEFAULT_INPUT_IMAGES_DIR,
    output_dir=DEFAULT_OUTPUT_DATASET_DIR,
    model_name="yolov8s-worldv2.pt",
    conf_thresh=CONFIDENCE_THRESHOLD
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure classes.txt exists in output directory
    classes_file = os.path.join(output_dir, "classes.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        for cls_name in CLASSES:
            f.write(f"{cls_name}\n")

    # Gather images
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in valid_exts:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_files:
        print(f"[WARNING] No image files found in: {input_dir}")
        return

    print(f"[INFO] Found {len(image_files)} images in '{input_dir}'.")
    print(f"[INFO] Loading model '{model_name}' onto {DEVICE_NAME}...")

    try:
        model = YOLO(model_name)
        if "world" in model_name:
            model.set_classes(list(PROMPT_TO_CLASS_ID.keys()))
            print("[INFO] Custom YOLO-World classes configured.")
    except Exception as e:
        print(f"[WARNING] Failed to load {model_name}, falling back to yolov8n.pt. Error: {e}")
        model = YOLO("yolov8n.pt")

    total_processed = 0
    total_saved = 0
    total_skipped = 0
    total_discarded = 0
    total_bboxes = 0

    for img_path in tqdm(image_files, desc="Auto-labeling image files"):
        img_stem = Path(img_path).stem
        dst_img_path = os.path.join(output_dir, f"{img_stem}.jpg")
        dst_txt_path = os.path.join(output_dir, f"{img_stem}.txt")

        # Auto-skip resume check
        if os.path.exists(dst_img_path) and os.path.exists(dst_txt_path):
            total_skipped += 1
            total_saved += 1
            continue

        results = model.predict(source=img_path, conf=conf_thresh, device=DEVICE, verbose=False)[0]
        img_h, img_w = results.orig_shape

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

        total_processed += 1

        if len(yolo_annotations) > 0:
            # Copy image to target dataset folder
            shutil.copy2(img_path, dst_img_path)
            
            # Write YOLO TXT label file
            with open(dst_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_annotations) + "\n")

            total_saved += 1
            total_bboxes += len(yolo_annotations)
        else:
            total_discarded += 1

    print("\n==========================================")
    print("IMAGE AUTO-LABELING & COLLECTION COMPLETE!")
    print("==========================================")
    print(f"Device Used               : {DEVICE_NAME}")
    print(f"Source Folder             : {input_dir}")
    print(f"Destination Dataset       : {output_dir}")
    print(f"New Images Processed      : {total_processed}")
    print(f"Skipped Images (Existing) : {total_skipped}")
    print(f"Total Images Saved        : {total_saved}")
    print(f"Discarded Images (Empty)  : {total_discarded}")
    print(f"Total Bounding Boxes      : {total_bboxes}")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Label External Image Directory into SEDIC Dataset")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_IMAGES_DIR, help="Source images directory")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DATASET_DIR, help="Target dataset directory")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    
    args = parser.parse_args()
    
    process_image_folder(
        input_dir=args.input,
        output_dir=args.output,
        conf_thresh=args.conf
    )
