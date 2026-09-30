import sys
import os
import shutil
import random
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
# SEDIC 2026 - YOLO MODEL TRAINING SCRIPT
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "Videos_Dataset")
YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

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

def prepare_train_val_split(val_ratio=0.2):
    """Ensures data.yaml and train/val indexing in Videos_Dataset are ready."""
    if os.path.exists(YAML_PATH) and os.path.exists(os.path.join(DATASET_DIR, "train.txt")):
        print(f"[INFO] Using Master YAML configuration at: {YAML_PATH}")
        return

    print("[INFO] Creating train/val split (80% Train, 20% Val)...")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Gather all image files directly under dataset/images/
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in os.listdir(IMAGES_DIR) 
        if os.path.isfile(os.path.join(IMAGES_DIR, f)) and Path(f).suffix.lower() in valid_exts
    ]

    random.seed(42)
    random.shuffle(image_files)

    num_val = int(len(image_files) * val_ratio)
    val_files = set(image_files[:num_val])
    train_files = set(image_files[num_val:])

    print(f"[INFO] Total images: {len(image_files)} -> Train: {len(train_files)}, Val: {len(val_files)}")

    for img_name in image_files:
        stem = Path(img_name).stem
        src_img = os.path.join(IMAGES_DIR, img_name)
        src_lbl = os.path.join(LABELS_DIR, f"{stem}.txt")

        if img_name in val_files:
            dst_img = os.path.join(val_img_dir, img_name)
            dst_lbl = os.path.join(val_lbl_dir, f"{stem}.txt")
        else:
            dst_img = os.path.join(train_img_dir, img_name)
            dst_lbl = os.path.join(train_lbl_dir, f"{stem}.txt")

        shutil.move(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)

    # Update data.yaml configuration
    clean_dataset_dir = DATASET_DIR.replace('\\', '/')
    yaml_content = f"""# SEDIC 2026 Dataset Configuration for YOLO Training
path: {clean_dataset_dir}
train: images/train
val: images/val

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

    print(f"[INFO] Updated configuration file: {YAML_PATH}")

def train_model(mode="fast", epochs=50, batch_size=16, model_size="n", imgsz=512, cache=False):
    """
    Trains YOLO model using Ultralytics optimized for GTX 1650 Ti (4GB VRAM) & Windows environment.
    
    Optimized Setup: 50 Epochs, Batch Size 16, Image Size 512px, YOLOv8 Nano, cache=False, amp=False, workers=2.
    """
    # Step 1: Ensure dataset split is ready
    prepare_train_val_split(val_ratio=0.2)

    # Step 2: Hardware check
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[HARDWARE] GPU Detected: {gpu_name}")
    else:
        device = 'cpu'
        print("[HARDWARE WARNING] No CUDA GPU detected. Training on CPU.")

    base_model = f"yolov8{model_size}.pt"
    print(f"\n⚡ [TRAINING STARTED (STABLE SPEED) - {epochs} EPOCHS / BATCH {batch_size}]")
    print(f" - Base Model   : {base_model}")
    print(f" - Image Size   : {imgsz}px")
    print(f" - Batch Size   : {batch_size}")
    print(f" - Epochs       : {epochs} (with Early Stopping patience=15)")
    print(f" - RAM Caching  : {cache}")

    model = YOLO(base_model)

    run_name = f"sedic_vessel_detector_{epochs}ep_b{batch_size}_{model_size}"

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
        patience=15,              # Early stopping if no improvement for 15 epochs
        workers=2,               # Reduced DataLoader workers for Windows spawn memory stability
        amp=False,               # Disabled AMP for GTX 1650 Ti (prevents NaN & AMP check failures)
        cache=cache              # Disabled RAM cache to prevent MemoryError (dataset is 39k+ images)
    )

    best_weights_path = os.path.join(BASE_DIR, "runs", "detect", run_name, "weights", "best.pt")

    print("\n=======================================================")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    print(f"Best Weights Saved At : {best_weights_path}")
    print("=======================================================")

    # Step 5: Automatically generate annotated video output
    print("\n🎬 Rendering full annotated video output using new trained AI model...")
    try:
        from predict import generate_video_output
        generate_video_output(video_source=os.path.join(BASE_DIR, "videos"))
    except Exception as e:
        print(f"[INFO] To render full video manually, run: python predict.py. Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Master Videos_Dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size (default: n)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size in pixels (default: 512)")

    args = parser.parse_args()
    train_model(
        mode="fast",
        epochs=args.epochs,
        batch_size=args.batch,
        model_size=args.model,
        imgsz=args.imgsz,
        cache=False
    )






