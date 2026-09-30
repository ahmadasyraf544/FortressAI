import os
import sys
import subprocess

# ==============================================================================
# SEDIC 2026 - LABELIMG LAUNCHER HELPER
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES_TXT = os.path.join(BASE_DIR, "classes.txt")

def list_and_choose_folder():
    print("="*70)
    print("       SEDIC 2026 - LABELIMG DATASET VIEWER / EDITOR")
    print("="*70)
    print("Pilih kumpulan dataset yang ingin dibuka:")
    print(" [1] Videos_Data2 (8 Folder Kelas Baharu)")
    print(" [2] Videos_Data1 (18 Folder Video)")
    print("="*70)
    
    choice = input("Pilihan Kumpulan [1 / 2, default: 1]: ").strip() or "1"
    
    if choice == "2":
        target_group = "Videos_Data1"
    else:
        target_group = "Videos_Data2"
        
    group_dir = os.path.join(BASE_DIR, "Videos_Dataset", target_group)
    if not os.path.exists(group_dir):
        print(f"[ERROR] Direktori tidak dijumpai: {group_dir}")
        return

    subfolders = sorted([f for f in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, f))])
    
    print(f"\nSenarai Folder dalam {target_group}:")
    for idx, f in enumerate(subfolders, 1):
        print(f" [{idx:02d}] {f}")
    print("="*70)
    
    sub_choice = input(f"Pilih Nombor Folder (1 - {len(subfolders)}) [default: 1]: ").strip() or "1"
    try:
        selected_idx = int(sub_choice) - 1
        if 0 <= selected_idx < len(subfolders):
            selected_folder = subfolders[selected_idx]
        else:
            selected_folder = subfolders[0]
    except ValueError:
        selected_folder = subfolders[0]

    folder_path = os.path.join(group_dir, selected_folder)
    images_dir = os.path.join(folder_path, "images")
    labels_dir = os.path.join(folder_path, "labels")
    classes_file = os.path.join(folder_path, "classes.txt")
    if not os.path.exists(classes_file):
        classes_file = CLASSES_TXT

    print(f"\n[INFO] Membuka LabelImg untuk: {selected_folder}")
    print(f"       Images : {images_dir}")
    print(f"       Labels : {labels_dir}")
    print(f"       Classes: {classes_file}\n")
    
    cmd = [
        "labelImg",
        images_dir,
        classes_file,
        labels_dir
    ]
    
    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        # Fallback running via python module
        subprocess.run([sys.executable, "-m", "labelImg", images_dir, classes_file, labels_dir])

if __name__ == "__main__":
    list_and_choose_folder()
