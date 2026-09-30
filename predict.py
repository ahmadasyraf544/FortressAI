import os
import glob
from ultralytics import YOLO

# ==============================================================================
# SEDIC 2026 - FULL VIDEO INFERENCE & RENDERING SCRIPT
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Search paths for trained weights (highest priority first)
WEIGHT_CANDIDATES = [
    os.path.join(BASE_DIR, "runs", "detect", "sedic_vessel_detector_50ep_b16_n", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "detect", "sedic_vessel_detector_150ep_b16", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "detect", "sedic_vessel_detector_fast", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "detect", "sedic_vessel_detector_high_acc", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "detect", "sedic_vessel_detector", "weights", "best.pt"),
    os.path.join(BASE_DIR, "yolov8s-worldv2.pt"),
    os.path.join(BASE_DIR, "yolov8s.pt")
]

def get_best_weights():
    for w in WEIGHT_CANDIDATES:
        if os.path.exists(w):
            return w
    return "yolov8s.pt"

def generate_video_output(video_source=None, conf=0.81):
    """
    Runs the trained AI model on video files, renders bounding boxes & class labels,
    and outputs full processed MP4/AVI videos.
    """
    if video_source is None:
        video_source = os.path.join(BASE_DIR, "videos")

    weights = get_best_weights()
    print(f"[INFO] Using model weights: {weights}")
    model = YOLO(weights)

    # Check if target is folder or single video
    if os.path.isdir(video_source):
        video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        videos = []
        for ext in video_exts:
            videos.extend(glob.glob(os.path.join(video_source, ext)))
        if not videos:
            print(f"[ERROR] No video files found in directory: {video_source}")
            return
        target_source = video_source
        print(f"[INFO] Found {len(videos)} videos in {video_source}. Processing...")
    else:
        target_source = video_source

    output_dir = os.path.join(BASE_DIR, "runs", "predict", "video_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🎬 [PROCESSING FULL VIDEO WITH AI DETECTOR]")
    print(f" - Input Video  : {video_source}")
    print(f" - Output Folder: {output_dir}")
    print(f" - Confidence   : {conf * 100}%\n")

    results = model.predict(
        source=target_source,
        conf=conf,
        save=True,               # Save full annotated video
        save_txt=False,          # Set True if you want text labels
        project=os.path.join(BASE_DIR, "runs", "predict"),
        name="video_results",
        exist_ok=True,
        stream=True              # Stream generator mode to prevent RAM memory accumulation
    )

    # Iterate stream generator to process and save video frames frame-by-frame
    for _ in results:
        pass

    print("\n=======================================================")
    print("VIDEO PROCESSING COMPLETED SUCCESSFULLY!")
    print("=======================================================")
    print(f"Full Annotated Videos Saved At: {output_dir}")
    print("=======================================================")

if __name__ == "__main__":
    generate_video_output(os.path.join(BASE_DIR, "videos"))
