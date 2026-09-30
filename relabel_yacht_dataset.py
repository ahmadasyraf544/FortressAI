import glob
import os
import re

TARGET_DIR = r"D:\SEDIC_2026\Videos_Dataset\Videos_Data1\01. yacht\labels"
YACHT_CLASS_ID = 4

lbl_files = sorted(glob.glob(os.path.join(TARGET_DIR, "yacht_frame_*.txt")))
print(f"[INFO] Found {len(lbl_files)} yacht label files to process.")

processed_count = 0
multi_box_resolved = 0
class_fixed_count = 0

for filepath in lbl_files:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        continue

    # Parse all bounding boxes: (cls_id, xc, yc, w, h, area)
    boxes = []
    for l in lines:
        parts = l.split()
        if len(parts) >= 5:
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            area = w * h
            boxes.append((cid, xc, yc, w, h, area))

    if not boxes:
        continue

    if len(boxes) > 1:
        multi_box_resolved += 1
        # If any box is already class 4, pick the largest class 4 box
        c4_boxes = [b for b in boxes if b[0] == YACHT_CLASS_ID]
        if c4_boxes:
            best_box = max(c4_boxes, key=lambda b: b[5])
        else:
            best_box = max(boxes, key=lambda b: b[5])
    else:
        best_box = boxes[0]

    # Check if class ID changed
    if best_box[0] != YACHT_CLASS_ID:
        class_fixed_count += 1

    # Overwrite file with single yacht label
    _, xc, yc, w, h, _ = best_box
    new_line = f"{YACHT_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_line)

    processed_count += 1

print("="*60)
print("RE-LABELING SUMMARY:")
print(f"Total files processed : {processed_count}")
print(f"Multi-boxes resolved  : {multi_box_resolved}")
print(f"Class IDs corrected   : {class_fixed_count}")
print("="*60)

# VERIFICATION STEP
verify_files = sorted(glob.glob(os.path.join(TARGET_DIR, "yacht_frame_*.txt")))
invalid_files = []
for vf in verify_files:
    with open(vf, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) != 1:
        invalid_files.append((vf, f"Line count {len(lines)} != 1"))
    else:
        parts = lines[0].split()
        if parts[0] != str(YACHT_CLASS_ID):
            invalid_files.append((vf, f"Class ID {parts[0]} != {YACHT_CLASS_ID}"))

if not invalid_files:
    print("VERIFICATION SUCCESSFUL: All 1,948 files have exactly ONE label of class '4' (yacht)!")
else:
    print(f"VERIFICATION FAILED for {len(invalid_files)} files: {invalid_files[:5]}")
