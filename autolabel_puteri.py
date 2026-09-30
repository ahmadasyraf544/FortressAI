import os
import glob
import re
import argparse
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# SEDIC 2026 - PUTERI INTAN SATU DATASET AUTO-LABELER & CLEANER
# ==============================================================================
# Target Video Frames: PUTERI INTAN SATU - MISC LNG tanker - binmei (1080p, h264)
# Range: frame_000050 to frame_008605
# Allowed Classes: Only 'tanker' (Class ID 1) and 'speedboat' (Class ID 5).
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
    "military_vessel_general"  # Class ID 9
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DEFAULT_PREFIX = "PUTERI INTAN SATU - MISC LNG tanker - binmei (1080p, h264)_frame_"
DEFAULT_MIN_FRAME = 50
DEFAULT_MAX_FRAME = 8605

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two bounding boxes (xc, yc, w, h)."""
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def process_puteri_intan_satu(
    dataset_dir=DEFAULT_DATASET_DIR,
    prefix=DEFAULT_PREFIX,
    min_frame=DEFAULT_MIN_FRAME,
    max_frame=DEFAULT_MAX_FRAME
):
    """
    Processes all PUTERI INTAN SATU dataset label files between min_frame and max_frame.
    Ensures that only 'tanker' (Class ID 1) and 'speedboat' (Class ID 5) annotations exist.
    """
    labels_dir = os.path.join(dataset_dir, "labels")

    if not os.path.exists(labels_dir):
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return

    # Find all label files matching prefix and within frame range
    all_label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    target_files = []

    for filepath in all_label_files:
        filename = Path(filepath).name
        if "classes.txt" in filename.lower():
            continue
        if prefix.lower() in filename.lower():
            match = re.search(r'frame_(\d+)', filename)
            if match:
                frame_num = int(match.group(1))
                if min_frame <= frame_num <= max_frame:
                    target_files.append((filepath, frame_num))

    target_files.sort(key=lambda x: x[1])

    if not target_files:
        print(f"[WARNING] No files found matching prefix '{prefix}' within frame range {min_frame} - {max_frame}.")
        return

    print(f"[INFO] Found {len(target_files)} label files to process for PUTERI INTAN SATU.")
    print(f"[INFO] Filtering classes to ONLY: 'tanker' (Class 1) and 'speedboat' (Class 5)...")

    total_files_updated = 0
    total_tanker_boxes = 0
    total_speedboat_boxes = 0
    total_discarded_boxes = 0

    for filepath, frame_num in tqdm(target_files, desc="Auto-labeling PUTERI INTAN SATU"):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        parsed_boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                area = w * h

                # Decision rule for PUTERI INTAN SATU clip:
                # The main LNG vessel is a tanker (Class ID 1).
                # Small escort / secondary vessel is speedboat (Class ID 5).
                if cid == 5 or (cid != 1 and area < 0.03):
                    new_cid = 5
                else:
                    new_cid = 1

                parsed_boxes.append((new_cid, xc, yc, w, h, area))

        # Sort by area descending so primary vessel takes precedence
        parsed_boxes.sort(key=lambda x: x[5], reverse=True)

        final_boxes = []
        for box in parsed_boxes:
            cid, xc, yc, w, h, area = box
            
            # Perform IoU deduplication per class
            overlap = False
            for f_cid, f_xc, f_yc, f_w, f_h in final_boxes:
                if f_cid == cid:
                    iou = calculate_iou((xc, yc, w, h), (f_xc, f_yc, f_w, f_h))
                    if iou > 0.4:
                        overlap = True
                        break

            if not overlap:
                final_boxes.append((cid, xc, yc, w, h))
            else:
                total_discarded_boxes += 1

        # Format lines for YOLO TXT file
        yolo_annotations = []
        for cid, xc, yc, w, h in final_boxes:
            if cid == 1:
                total_tanker_boxes += 1
            elif cid == 5:
                total_speedboat_boxes += 1

            yolo_annotations.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # Overwrite label file with clean annotations
        with open(filepath, "w", encoding="utf-8") as f:
            if yolo_annotations:
                f.write("\n".join(yolo_annotations) + "\n")
            else:
                f.write("")

        total_files_updated += 1

    print("\n=======================================================")
    print("PUTERI INTAN SATU AUTO-LABELING COMPLETED!")
    print("=======================================================")
    print(f"Dataset Directory         : {dataset_dir}")
    print(f"Total Files Processed     : {total_files_updated}")
    print(f"Frame Range               : {min_frame} - {max_frame}")
    print(f"Total 'tanker' Boxes (1)   : {total_tanker_boxes}")
    print(f"Total 'speedboat' Boxes (5): {total_speedboat_boxes}")
    print(f"Duplicate Boxes Removed   : {total_discarded_boxes}")
    print("=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-label and filter PUTERI INTAN SATU dataset to tanker and speedboat classes only."
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_DIR, help="Target dataset directory")
    parser.add_argument("--min-frame", type=int, default=DEFAULT_MIN_FRAME, help="Minimum frame number")
    parser.add_argument("--max-frame", type=int, default=DEFAULT_MAX_FRAME, help="Maximum frame number")
    
    args = parser.parse_args()

    process_puteri_intan_satu(
        dataset_dir=args.dataset,
        min_frame=args.min_frame,
        max_frame=args.max_frame
    )
