import os
import cv2
import glob
from pathlib import Path

# ==============================================================================
# PREVIEW AUTO-LABELED DATASET (SEDIC 2026)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "Videos_Dataset")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.txt")

# 1. Load class names from classes.txt
if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        CLASSES = [line.strip() for line in f.readlines() if line.strip()]
else:
    CLASSES = [
        "container_ship", "tanker", "cargo_ship", "passenger_ferry", "yacht",
        "speedboat", "fishing_boat", "military_vessel_local",
        "military_vessel_foreign", "military_vessel_general",
        "military_helicopter", "military_airfighter", "military_submarine",
        "tug_boat"
    ]

# Bounding box color palette (BGR)
COLORS = [
    (0, 255, 0),      # 0: container_ship (Green)
    (255, 0, 0),      # 1: tanker (Blue)
    (0, 165, 255),    # 2: cargo_ship (Orange)
    (255, 0, 255),    # 3: passenger_ferry (Magenta)
    (0, 255, 255),    # 4: yacht (Yellow)
    (255, 255, 0),    # 5: speedboat (Cyan)
    (0, 128, 255),    # 6: fishing_boat (Bright Orange)
    (128, 0, 128),    # 7: military_vessel_local (Dark Purple)
    (0, 0, 255),      # 8: military_vessel_foreign (Red)
    (128, 128, 128),  # 9: military_vessel_general (Grey)
    (255, 140, 0),    # 10: military_helicopter (Deep Sky Blue / Gold)
    (50, 205, 50),    # 11: military_airfighter (Lime Green)
    (180, 105, 255),  # 12: military_submarine (Hot Pink / Magenta)
    (22, 204, 132)    # 13: tug_boat (Tactical Lime)
]

def main():
    # Look for images in Videos_Dataset/ recursively
    image_paths = []
    if os.path.exists(DATASET_DIR):
        image_paths.extend(glob.glob(os.path.join(DATASET_DIR, "**", "*.jpg"), recursive=True))
        image_paths.extend(glob.glob(os.path.join(DATASET_DIR, "**", "*.png"), recursive=True))

    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"[ERROR] No image files found in {DATASET_DIR}.")
        return

    print("=================================================================")
    print(f"FOUND {len(image_paths):,} IMAGES IN VIDEOS_DATASET.")
    print("KEYBOARD CONTROLS FOR PREVIEW WINDOW:")
    print(" - Press 'D' or 'Space' or ANY KEY : NEXT IMAGE")
    print(" - Press 'A'                        : PREVIOUS IMAGE")
    print(" - Press 'Q' or 'ESC'              : QUIT PREVIEW")
    print("=================================================================")

    index = 0
    while 0 <= index < len(image_paths):
        img_path = image_paths[index]
        img_stem = Path(img_path).stem
        img_parent = Path(img_path).parent

        # Search label in labels/ sibling folder or same folder
        candidate_lbl1 = os.path.join(img_parent.parent, "labels", f"{img_stem}.txt")
        candidate_lbl2 = os.path.join(img_parent, f"{img_stem}.txt")

        if os.path.exists(candidate_lbl1):
            txt_path = candidate_lbl1
        elif os.path.exists(candidate_lbl2):
            txt_path = candidate_lbl2
        else:
            txt_path = None

        img = cv2.imread(img_path)
        if img is None:
            index += 1
            continue

        img_h, img_w, _ = img.shape

        # Read YOLO label file if exists
        labels_count = 0
        if txt_path and os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])

                    x1 = int((xc - w / 2.0) * img_w)
                    y1 = int((yc - h / 2.0) * img_h)
                    x2 = int((xc + w / 2.0) * img_w)
                    y2 = int((yc + h / 2.0) * img_h)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)

                    cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class_{cls_id}"
                    color = COLORS[cls_id % len(COLORS)]

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{cls_id}: {cls_name}"
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
                    cv2.putText(img, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    labels_count += 1

        info_text = f"[{index + 1}/{len(image_paths)}] {Path(img_path).name} | Objects: {labels_count}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        win_name = "SEDIC 2026 - Dataset Preview (D: Next, A: Prev, Q: Quit)"
        cv2.imshow(win_name, img)

        key = cv2.waitKey(0) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key in [ord('a'), ord('A')]:
            index = max(0, index - 1)
        else:
            index += 1

    cv2.destroyAllWindows()
    print("[INFO] Preview window closed.")

if __name__ == "__main__":
    main()
