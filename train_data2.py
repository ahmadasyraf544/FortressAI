import sys
import os
import random
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ==============================================================================
# SEDIC 2026 - TRAINING SCRIPT FOR VIDEOS_DATA2 DATASET ONLY
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA2_DIR = os.path.join(BASE_DIR, "Videos_Dataset", "Videos_Data2")
YAML_PATH = os.path.join(DATA2_DIR, "data.yaml")
TRAIN_TXT = os.path.join(DATA2_DIR, "train.txt")
VAL_TXT = os.path.join(DATA2_DIR, "val.txt")

CLASSES = [
    "container_ship",
    "tanker",
    "cargo_ship",
    "passenger_ferry",
    "yacht",
    "speedboat",
    "fishing_boat",
    "military_vessel_local",
    "military_vessel_foreign",
    "military_vessel_general",
    "military_helicopter",
    "military_airfighter",
    "military_submarine",
    "tug_boat"
]

def prepare_data2_split(val_ratio=0.2):
    """
    Scans all image files within Videos_Data2 subfolders and generates
    train.txt and val.txt with 80% Train and 20% Validation split.
    """
    if os.path.exists(TRAIN_TXT) and os.path.exists(VAL_TXT) and os.path.exists(YAML_PATH):
        print(f"[INFO] Using existing train.txt & val.txt in: {DATA2_DIR}")
        return

    print(f"\n[INFO] Indexing dataset images in {DATA2_DIR}...")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = []

    for root, dirs, files in os.walk(DATA2_DIR):
        if os.path.basename(root).lower() == "images":
            for f in files:
                if Path(f).suffix.lower() in valid_exts and f.lower() != "classes.txt":
                    # Check matching label
                    img_path = os.path.join(root, f)
                    lbl_path = img_path.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
                    lbl_path = os.path.splitext(lbl_path)[0] + ".txt"
                    if os.path.exists(lbl_path):
                        all_images.append(img_path)

    if not all_images:
        raise FileNotFoundError(f"[ERROR] No valid images with matching labels found in: {DATA2_DIR}")

    random.seed(42)
    random.shuffle(all_images)

    num_val = int(len(all_images) * val_ratio)
    val_list = all_images[:num_val]
    train_list = all_images[num_val:]

    print(f"[INFO] Total valid images in Videos_Data2: {len(all_images):,}")
    print(f"       -> Train Set (80%): {len(train_list):,} images")
    print(f"       -> Val Set   (20%): {len(val_list):,} images")

    # Write train.txt
    with open(TRAIN_TXT, "w", encoding="utf-8") as f:
        for p in train_list:
            f.write(f"{p}\n")

    # Write val.txt
    with open(VAL_TXT, "w", encoding="utf-8") as f:
        for p in val_list:
            f.write(f"{p}\n")

    # Write data.yaml
    clean_data2_path = DATA2_DIR.replace('\\', '/')
    yaml_content = f"""# SEDIC 2026 Dataset - Videos_Data2 Only
path: {clean_data2_path}
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
  10: military_helicopter
  11: military_airfighter
  12: military_submarine
  13: tug_boat
"""
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[INFO] Generated dataset config: {YAML_PATH}\n")


def train_data2(epochs=50, batch_size=16, model_size="n", imgsz=512, patience=15):
    """
    Executes YOLOv8 model training exclusively on Videos_Data2.
    Runs in foreground with real-time interactive terminal logs.
    """
    prepare_data2_split(val_ratio=0.2)

    # Check CUDA GPU
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 [HARDWARE] GPU Detected: {gpu_name} (CUDA:0)")
    else:
        device = "cpu"
        print("⚠️ [HARDWARE WARNING] No CUDA GPU detected. Training on CPU.")

    base_model = f"yolov8{model_size}.pt"
    run_name = f"sedic_vessel_detector_data2_{epochs}ep_b{batch_size}_{model_size}"

    print("=" * 70)
    print(f"⚡ [STARTING YOLOV8 TRAINING - VIDEOS_DATA2 ONLY]")
    print(f" - Dataset Path : {DATA2_DIR}")
    print(f" - Config YAML  : {YAML_PATH}")
    print(f" - Base Model   : {base_model}")
    print(f" - Epochs       : {epochs} (Early Stopping patience: {patience})")
    print(f" - Batch Size   : {batch_size}")
    print(f" - Image Size   : {imgsz}px")
    print(f" - Device       : {device}")
    print("=" * 70)

    model = YOLO(base_model)

    results = model.train(
        data=YAML_PATH,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=os.path.join(BASE_DIR, "runs", "detect"),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        patience=patience,
        workers=2,
        amp=False,          # Set False for GTX 1650 Ti stability
        cache=False         # Disabled RAM cache to prevent OutOfMemory
    )

    best_weights_path = os.path.join(BASE_DIR, "runs", "detect", run_name, "weights", "best.pt")

    print("\n" + "=" * 70)
    print("🎉 TRAINING ON VIDEOS_DATA2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"🏆 Best Weights Saved At : {best_weights_path}")
    print("=" * 70)
    print("\nTip: Anda boleh gunakan weights ini dalam GUI dengan meletakkannya di config.py atau predict.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Videos_Data2 Dataset Only")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size (default: n)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size in pixels (default: 512)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")

    args = parser.parse_args()

    train_data2(
        epochs=args.epochs,
        batch_size=args.batch,
        model_size=args.model,
        imgsz=args.imgsz,
        patience=args.patience
    )
