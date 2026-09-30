import os
import shutil
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_SEPARATED_DIR = os.path.join(BASE_DIR, "Videos_Dataset", "Videos_Data1")

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

VIDEO_TARGETS = [
    (1, "Avante_Yacht_Charter_Langkawi_DJI_Mavic_Air_2_Amirul_Azrai_1080p"),
    (2, "Container_Ship《MTT_SAISUNEE》Crossing_the_Ma_Wan_Channel_in_HK_20260529"),
    (3, "Drone in malacca strait - Q Cruz (1080p, h264)"),
    (4, "Ferry_Terminal_Georgetown_Drone_View,_Penang_NSTOS_480p,_h264"),
    (5, "Hari_Terbuka_Armada,_Pangkalan_TLDM,_Lumut,_Perak,_Malaysia_Basic"),
    (6, "KD Rahmat Museum - Awan Media (1080p, h264)"),
    (7, "KD_KASTURI_REFIT_Program_Lumut_Naval_Shipyard_LUNAS_Malaysia_Lumut"),
    (8, "KD_Lekir_F_26_One_of_the_Pride_Corvettes_of_the_Malaysian_Government"),
    (9, "KM ARAU - Berjaya Dockyard (1080p, h264)"),
    (10, "KM Pekan Aerial Demo Test HD - Tderr Tech (1080p, h264)"),
    (11, "KMBaganDatuk - Eddie Putera (1080p, h264)"),
    (12, "kuala kedah on top view - Amir Nazmir Channel (1080p, h264)"),
    (13, "LIMA_2025_Maritime_Demonstration_Pirates_Helicopter_Police_Navy"),
    (14, "Media_Footage_LNG_ship_the_'Seri_Bakti'_arrives_at_Santos_GLNG_site"),
    (15, "Orkim_Safety_Video_Orkim_Power_Orkim_Group_of_Companies_1080p,_h264"),
    (16, "Penang Island Ferry - BarneySuperBear (480p, h264)"),
    (17, "Port_Adelaide_Shipping_95_Zenith_Leader_Rx224_720p,_h264"),
    (18, "PUTERI INTAN SATU - MISC LNG tanker - binmei (1080p, h264)")
]

def find_all_files():
    print("[INFO] Indexing all images and labels in dataset/...")
    all_imgs = glob.glob(os.path.join(DATASET_DIR, "images", "**", "*.jpg"), recursive=True) + \
               glob.glob(os.path.join(DATASET_DIR, "images", "**", "*.png"), recursive=True)
    all_labels = glob.glob(os.path.join(DATASET_DIR, "labels", "**", "*.txt"), recursive=True)
    
    img_dict = {}
    for p in all_imgs:
        stem = Path(p).stem
        img_dict[stem] = p
        
    lbl_dict = {}
    for p in all_labels:
        if Path(p).name.lower() == "classes.txt":
            continue
        stem = Path(p).stem
        lbl_dict[stem] = p
        
    print(f"[INFO] Found {len(img_dict)} unique images and {len(lbl_dict)} unique labels.")
    return img_dict, lbl_dict

def link_or_copy_file(src, dst):
    if not os.path.exists(dst):
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)

def separate_dataset():
    img_dict, lbl_dict = find_all_files()
    
    os.makedirs(OUTPUT_SEPARATED_DIR, exist_ok=True)
    
    assigned_images = set()
    summary_stats = []
    
    for idx, vname in VIDEO_TARGETS:
        folder_name = f"{idx:02d}. {vname}"
        target_dir = os.path.join(OUTPUT_SEPARATED_DIR, folder_name)
        target_imgs_dir = os.path.join(target_dir, "images")
        target_lbls_dir = os.path.join(target_dir, "labels")
        
        os.makedirs(target_imgs_dir, exist_ok=True)
        os.makedirs(target_lbls_dir, exist_ok=True)
        
        # Write classes.txt
        classes_txt_path = os.path.join(target_dir, "classes.txt")
        classes_txt_lbls_path = os.path.join(target_lbls_dir, "classes.txt")
        with open(classes_txt_path, "w", encoding="utf-8") as f:
            for c in CLASSES:
                f.write(f"{c}\n")
        shutil.copy2(classes_txt_path, classes_txt_lbls_path)
        
        # Write data.yaml
        abs_target_dir = os.path.abspath(target_dir).replace('\\', '/')
        yaml_content = f"""# SEDIC 2026 Dataset - {vname}
path: {abs_target_dir}
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
        with open(os.path.join(target_dir, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_content)
            
        v_clean = vname.replace("'", "_").replace("《", "_").replace("》", "_").replace(",", "_").replace(" ", "_")
        
        # Find matching stems
        matching_stems = [
            stem for stem in img_dict 
            if vname.lower() in stem.lower() or v_clean.lower() in stem.lower() or stem.lower().startswith(vname[:15].lower())
        ]
        
        tasks = []
        for stem in matching_stems:
            assigned_images.add(stem)
            img_src = img_dict[stem]
            img_dst = os.path.join(target_imgs_dir, Path(img_src).name)
            tasks.append((img_src, img_dst))
            
            if stem in lbl_dict:
                lbl_src = lbl_dict[stem]
                lbl_dst = os.path.join(target_lbls_dir, Path(lbl_src).name)
                tasks.append((lbl_src, lbl_dst))
                
        # Perform linking/copying in parallel
        with ThreadPoolExecutor(max_workers=16) as executor:
            list(tqdm(executor.map(lambda pair: link_or_copy_file(pair[0], pair[1]), tasks), 
                      total=len(tasks), 
                      desc=f"[{idx:02d}/18] {vname[:30]}...", 
                      leave=False))
            
        img_count = len(os.listdir(target_imgs_dir))
        lbl_count = len([f for f in os.listdir(target_lbls_dir) if f.endswith(".txt") and f != "classes.txt"])
        
        summary_stats.append({
            "index": idx,
            "name": vname,
            "images": img_count,
            "labels": lbl_count
        })
        print(f"[{idx:02d}/18] {vname} -> {img_count} images, {lbl_count} labels")

    # Handle remaining / SMD benchmark images
    unassigned = [stem for stem in img_dict if stem not in assigned_images]
    if unassigned:
        smd_dir = os.path.join(OUTPUT_SEPARATED_DIR, "19. Singapore_Maritime_Dataset_Benchmark")
        smd_imgs_dir = os.path.join(smd_dir, "images")
        smd_lbls_dir = os.path.join(smd_dir, "labels")
        os.makedirs(smd_imgs_dir, exist_ok=True)
        os.makedirs(smd_lbls_dir, exist_ok=True)
        
        tasks = []
        for stem in unassigned:
            img_src = img_dict[stem]
            tasks.append((img_src, os.path.join(smd_imgs_dir, Path(img_src).name)))
            if stem in lbl_dict:
                lbl_src = lbl_dict[stem]
                tasks.append((lbl_src, os.path.join(smd_lbls_dir, Path(lbl_src).name)))
                
        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(lambda pair: link_or_copy_file(pair[0], pair[1]), tasks))
            
        print(f"[19/19] SMD Maritime Benchmark -> {len(os.listdir(smd_imgs_dir))} images, {len(os.listdir(smd_lbls_dir))} labels")

    print("\n" + "="*80)
    print("DATASET SEPARATION COMPLETED SUCCESSFULLY!")
    print(f"Output Directory: {OUTPUT_SEPARATED_DIR}")
    print("="*80)
    for s in summary_stats:
        print(f"{s['index']:02d}. {s['name']} -> {s['images']:,} images | {s['labels']:,} labels")
    print("="*80)

if __name__ == "__main__":
    separate_dataset()
