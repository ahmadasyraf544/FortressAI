import os
import glob
from pathlib import Path

# ==============================================================================
# SEDIC 2026 - LABEL REPAIR SCRIPT FOR MTT_SAISUNEE CONTAINER SHIP
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TARGET_PREFIX = "Container_Ship《MTT_SAISUNEE》"

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two YOLO boxes [xc, yc, w, h]."""
    b1_x1 = box1[0] - box1[2] / 2.0
    b1_y1 = box1[1] - box1[3] / 2.0
    b1_x2 = box1[0] + box1[2] / 2.0
    b1_y2 = box1[1] + box1[3] / 2.0

    b2_x1 = box2[0] - box2[2] / 2.0
    b2_y1 = box2[1] - box2[3] / 2.0
    b2_x2 = box2[0] + box2[2] / 2.0
    b2_y2 = box2[1] + box2[3] / 2.0

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def fix_mtt_saisunee_labels():
    # Find all txt files matching MTT_SAISUNEE in dataset
    all_label_files = glob.glob(os.path.join(DATASET_DIR, "**", "*MTT_SAISUNEE*.txt"), recursive=True)
    
    if not all_label_files:
        print(f"[ERROR] No label files found matching: {TARGET_PREFIX}")
        return

    print(f"[INFO] Found {len(all_label_files)} label files for MTT_SAISUNEE")
    print("[INFO] Cleaning duplicate boxes & updating class to 0 (container_ship)...")

    fixed_count = 0
    total_boxes_before = 0
    total_boxes_after = 0

    for file_path in all_label_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    boxes.append((cls_id, xc, yc, w, h))

            if not boxes:
                continue

            total_boxes_before += len(boxes)

            # Step 1: Reassign class ID for container ship
            # Main ship in MTT_SAISUNEE video is container_ship (class 0)
            updated_boxes = []
            for cls_id, xc, yc, w, h in boxes:
                area = w * h
                # If box is main vessel (area > 0.03 or width > 0.2 or height > 0.2 or mislabeled class)
                if area > 0.03 or w > 0.25 or h > 0.25 or cls_id in (4, 9, 7, 2, 1, 3):
                    new_cls = 0  # container_ship
                else:
                    new_cls = cls_id
                updated_boxes.append((new_cls, xc, yc, w, h))

            # Step 2: NMS Deduplication (remove overlapping duplicate boxes for same ship)
            dedup_boxes = []
            for item in updated_boxes:
                cls_id, xc, yc, w, h = item
                box_vec = [xc, yc, w, h]
                duplicate = False
                for d_cls, d_xc, d_yc, d_w, d_h in dedup_boxes:
                    if calculate_iou(box_vec, [d_xc, d_yc, d_w, d_h]) > 0.5:
                        duplicate = True
                        break
                if not duplicate:
                    dedup_boxes.append((cls_id, xc, yc, w, h))

            total_boxes_after += len(dedup_boxes)

            # Format new label text
            new_lines = [
                f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                for cls_id, xc, yc, w, h in dedup_boxes
            ]

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            fixed_count += 1

        except Exception as e:
            print(f"[WARNING] Error processing {file_path}: {e}")

    print("\n=======================================================")
    print("LABEL REPAIR COMPLETED!")
    print("=======================================================")
    print(f"Total Files Processed   : {fixed_count}")
    print(f"Total Bounding Boxes    : {total_boxes_before} -> {total_boxes_after} (Deduplicated)")
    print(f"Container Ship Class ID : 0 (container_ship)")
    print("=======================================================")

if __name__ == "__main__":
    fix_mtt_saisunee_labels()
