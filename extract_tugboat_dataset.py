import os
import sys
import cv2
import torch
import random
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# SEDIC 2026 - GENERATE DATASET FOR TUG BOAT (VIDEOS_DATA2)
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "Videos_Data2", "tug boat.mp4")
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "Videos_Dataset", "Videos_Data2", "09. tug boat")
IMGS_DIR = os.path.join(OUTPUT_DATASET_DIR, "images")
LBLS_DIR = os.path.join(OUTPUT_DATASET_DIR, "labels")

CLASSES_FILE = os.path.join(BASE_DIR, "classes.txt")
with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    CLASSES = [line.strip() for line in f if line.strip()]

TARGET_CLASS_ID = 13  # tug_boat
TARGET_FPS = 3.0
CONF_THRESH = 0.18

def main():
    print("=" * 70)
    print("🚢 [SEDIC 2026] GENERATING TUG BOAT DATASET FOR VIDEOS_DATA2")
    print(f" - Input Video  : {VIDEO_PATH}")
    print(f" - Output Folder: {OUTPUT_DATASET_DIR}")
    print(f" - Target Class : ID {TARGET_CLASS_ID} ({CLASSES[TARGET_CLASS_ID]})")
    print(f" - Sampling Rate: {TARGET_FPS} FPS")
    print("=" * 70)

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        return

    os.makedirs(IMGS_DIR, exist_ok=True)
    os.makedirs(LBLS_DIR, exist_ok=True)

    # Write classes.txt
    classes_content = "\n".join(CLASSES) + "\n"
    with open(os.path.join(OUTPUT_DATASET_DIR, "classes.txt"), "w", encoding="utf-8") as f:
        f.write(classes_content)
    with open(os.path.join(LBLS_DIR, "classes.txt"), "w", encoding="utf-8") as f:
        f.write(classes_content)

    # Write data.yaml for 09. tug boat
    abs_vdir = os.path.abspath(OUTPUT_DATASET_DIR).replace('\\', '/')
    yaml_lines = [
        f"# SEDIC 2026 Dataset - Videos_Data2: 09. tug boat",
        f"path: {abs_vdir}",
        f"train: images",
        f"val: images",
        "",
        "names:"
    ]
    for cid, cname in enumerate(CLASSES):
        yaml_lines.append(f"  {cid}: {cname}")

    with open(os.path.join(OUTPUT_DATASET_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    # Load YOLO detection model
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[HARDWARE] Using device: {device} ({torch.cuda.get_device_name(0) if device==0 else 'CPU'})")

    if os.path.exists(os.path.join(BASE_DIR, "yolov8s-worldv2.pt")):
        model_path = os.path.join(BASE_DIR, "yolov8s-worldv2.pt")
        model = YOLO(model_path)
        try:
            model.set_classes(["tugboat", "tug boat", "boat", "vessel", "ship"])
            print("[INFO] YOLO-World custom prompt classes set: ['tugboat', 'tug boat', 'boat', 'vessel', 'ship']")
        except Exception as e:
            print(f"[WARNING] Could not set custom classes on world model: {e}")
    elif os.path.exists(os.path.join(BASE_DIR, "yolov8m.pt")):
        model = YOLO(os.path.join(BASE_DIR, "yolov8m.pt"))
    elif os.path.exists(os.path.join(BASE_DIR, "yolov8s.pt")):
        model = YOLO(os.path.join(BASE_DIR, "yolov8s.pt"))
    else:
        model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / TARGET_FPS))

    print(f"[INFO] Video Stats: {total_vid_frames:,} total frames @ {fps:.2f} FPS. Sampling every {step} frames.")

    frame_count = 0
    saved_count = 0
    discarded_count = 0

    with tqdm(total=total_vid_frames, desc="Extracting Tug Boat", unit="frames") as pbar:
        while cap.isOpened() and frame_count < total_vid_frames:
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

            base_filename = f"tug_boat_frame_{frame_count:06d}"
            img_path = os.path.join(IMGS_DIR, f"{base_filename}.jpg")
            txt_path = os.path.join(LBLS_DIR, f"{base_filename}.txt")

            img_h, img_w, _ = frame.shape

            # YOLO inference
            results = model.predict(
                source=frame,
                conf=CONF_THRESH,
                device=device,
                verbose=False
            )[0]

            annotations = []
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
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

                    annotations.append(f"{TARGET_CLASS_ID} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            if len(annotations) > 0:
                cv2.imwrite(img_path, frame)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(annotations) + "\n")
                saved_count += 1
            else:
                discarded_count += 1

    cap.release()

    print("\n" + "=" * 70)
    print(f"✅ TUG BOAT DATASET EXTRACTION COMPLETED!")
    print(f" - Saved Frames & Labels: {saved_count:,} images")
    print(f" - Discarded (no target) : {discarded_count:,} frames")
    print(f" - Dataset Folder        : {OUTPUT_DATASET_DIR}")
    print("=" * 70)

    # Update train.txt and val.txt in Videos_Data2
    data2_dir = os.path.join(BASE_DIR, "Videos_Dataset", "Videos_Data2")
    all_data2_images = []
    for root, dirs, files in os.walk(data2_dir):
        if os.path.basename(root).lower() == "images":
            for f in files:
                if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"} and f.lower() != "classes.txt":
                    img_p = os.path.join(root, f)
                    lbl_p = img_p.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
                    lbl_p = os.path.splitext(lbl_p)[0] + ".txt"
                    if os.path.exists(lbl_p):
                        all_data2_images.append(img_p)

    random.seed(42)
    random.shuffle(all_data2_images)
    split_idx = int(len(all_data2_images) * 0.8)

    train_data2_txt = os.path.join(data2_dir, "train.txt")
    val_data2_txt = os.path.join(data2_dir, "val.txt")

    with open(train_data2_txt, "w", encoding="utf-8") as f:
        for p in all_data2_images[:split_idx]:
            f.write(f"{os.path.abspath(p)}\n")

    with open(val_data2_txt, "w", encoding="utf-8") as f:
        for p in all_data2_images[split_idx:]:
            f.write(f"{os.path.abspath(p)}\n")

    print(f"[INFO] Updated Videos_Data2 Dataset Split:")
    print(f"       -> Total Images : {len(all_data2_images):,}")
    print(f"       -> Train Set    : {split_idx:,}")
    print(f"       -> Val Set      : {len(all_data2_images) - split_idx:,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
