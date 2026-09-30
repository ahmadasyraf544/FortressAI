# SEDIC 2026 - Setup & Annotation Guide for Team Members

This document contains everything you need to download, install, and set up on your computer to perform image annotation and label verification for the SEDIC 2026 dataset.

---

## 📥 Required Downloads & Software

Before starting, make sure you have the following software installed on your Windows machine:

1. **VS Code (Code Editor)**:
   - Download & Install from: [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. **Python (Version 3.9 - 3.11)**:
   - Download & Install from: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - ⚠️ **Important during Python installation**: Check the box **"Add Python to PATH"** before clicking Install.

---

## ⚙️ Step-by-Step Installation Commands

Follow these steps to set up the environment and install **LabelImg**:

### 1. Open VS Code & Open Project Folder
- Open **VS Code** $\rightarrow$ Click **File** $\rightarrow$ **Open Folder...** $\rightarrow$ Select `D:\SEDIC_2026`.

### 2. Open Integrated Terminal
- Press **`Ctrl` + `~`** (or `Ctrl` + `Shift` + `` ` ``) in VS Code to open the terminal.

### 3. Create & Activate Virtual Environment (Recommended)
Copy and paste these commands into your VS Code terminal:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```
*(You will see `(venv)` appear at the front of your terminal prompt).*

### 4. Install All Required Python Libraries
Copy and paste this command into the terminal:

```powershell
pip install labelImg opencv-python ultralytics tqdm
```

---

## 🚀 How to Launch LabelImg for Dataset Review

Once installation is complete, launch **LabelImg** by running:

```powershell
labelImg D:\SEDIC_2026\dataset\images D:\SEDIC_2026\dataset\classes.txt D:\SEDIC_2026\dataset\labels
```

### ⚙️ GUI Verification Check:
When LabelImg opens:
1. **Open Dir**: Should point to `D:\SEDIC_2026\dataset\images`
2. **Change Save Dir**: Should point to `D:\SEDIC_2026\dataset\labels`
3. **Format Toggle**: Must be set to **`YOLO`** (Click the button if it displays `PascalVOC` until it changes to `YOLO`).
4. Press **`D`** on your keyboard to start browsing labeled images!

---

## 👁️ Visual Preview Tool (`preview.py`)

To quickly preview images with colored bounding boxes on your screen without opening LabelImg:

```powershell
python preview.py
```
- Press **`D`** or **`Space`**: Next Image
- Press **`A`**: Previous Image
- Press **`Q`** or **`ESC`**: Exit Preview

---

## 🏷️ Ship Class Reference Guide (10 Target Classes)

| Class ID | Class Name | Description |
| :---: | :--- | :--- |
| `0` | `container_ship` | Commercial vessels carrying stacked shipping containers. |
| `1` | `tanker` | Oil/chemical tankers with flat decks, pipes, or liquid tanks. |
| `2` | `cargo_ship` | Bulk carriers, freighters, or general cargo vessels. |
| `3` | `passenger_ferry` | Passenger ferries, cruise ships, or passenger watercraft. |
| `4` | `yacht` | Commercial or luxury motor/sailing yachts. |
| `5` | `speedboat` | Small high-speed motorboats, RHIBs, or jet boats. |
| `6` | `fishing_boat` | Traditional or industrial fishing trawlers, sampans, or boats. |
| `7` | `military_vessel_local` | Domestic naval/coast guard vessels (grey hull/pennant numbers). |
| `8` | `military_vessel_foreign` | Foreign warships, destroyers, frigates, or aircraft carriers. |
| `9` | `military_vessel_general` | Unidentified or general naval military vessels. |

---

## ⌨️ LabelImg Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| **`D`** | **Next Image** |
| **`A`** | **Previous Image** |
| **`W`** | Draw **New Bounding Box** |
| **`Del`** | **Delete** selected box |
| **`Ctrl + S`** | **Save** annotation changes |
