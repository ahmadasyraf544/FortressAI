import os
import shutil
import glob
from pathlib import Path

# ==============================================================================
# SEDIC 2026 - DATASET REORGANIZER FOR YOLO & LABELIMG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

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
    "military_vessel_general"
]

def organize_dataset():
    """
    Reorganizes a flat dataset directory into standard YOLO format:
    dataset/
      ├── images/        <-- All .jpg / .png image files
      ├── labels/        <-- All .txt label annotation files
      ├── classes.txt    <-- Class index file for LabelImg
      └── data.yaml      <-- YOLO model training configuration file
    """
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        return

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    print(f"[INFO] Organizing dataset in: {DATASET_DIR}")

    # Gather loose image files directly under dataset/ root
    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(DATASET_DIR, ext)))
        image_files.extend(glob.glob(os.path.join(DATASET_DIR, ext.upper())))

    # Gather loose label files directly under dataset/ root
    label_files = [
        f for f in glob.glob(os.path.join(DATASET_DIR, "*.txt"))
        if Path(f).name.lower() != "classes.txt"
    ]

    print(f"[INFO] Found {len(image_files)} loose images and {len(label_files)} loose labels to organize.")

    # Move images to dataset/images/
    moved_images = 0
    for img_file in image_files:
        if os.path.exists(img_file):
            dst = os.path.join(IMAGES_DIR, Path(img_file).name)
            try:
                shutil.move(img_file, dst)
                moved_images += 1
            except Exception:
                pass

    # Move label txt files to dataset/labels/
    moved_labels = 0
    for lbl_file in label_files:
        if os.path.exists(lbl_file):
            dst = os.path.join(LABELS_DIR, Path(lbl_file).name)
            try:
                shutil.move(lbl_file, dst)
                moved_labels += 1
            except Exception:
                pass

    # Generate / update classes.txt in root and labels directory
    classes_root = os.path.join(DATASET_DIR, "classes.txt")
    classes_labels = os.path.join(LABELS_DIR, "classes.txt")

    with open(classes_root, "w", encoding="utf-8") as f:
        for cls_name in CLASSES:
            f.write(f"{cls_name}\n")
    
    shutil.copy2(classes_root, classes_labels)

    # Generate data.yaml for YOLO training
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    yaml_content = f"""# SEDIC 2026 Dataset Configuration for YOLO Training
path: {DATASET_DIR}
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

    print("\n=======================================================")
    print("DATASET REORGANIZATION COMPLETE!")
    print("=======================================================")
    print(f"Images in 'dataset/images/' : {len(os.listdir(IMAGES_DIR))}")
    print(f"Labels in 'dataset/labels/' : {len(os.listdir(LABELS_DIR))}")
    print(f"Class definitions file      : {classes_root}")
    print(f"YOLO Training Config         : {yaml_path}")
    print("=======================================================")

if __name__ == "__main__":
    organize_dataset()
