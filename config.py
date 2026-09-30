import os
from pathlib import Path

# ==============================================================================
# SEDIC 2026 - TACTICAL MARITIME SURVEILLANCE HUD: CONFIGURATION
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths
DEFAULT_WEIGHTS_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "sedic_vessel_detector_data2_150ep_b5_n", "weights", "best.pt"
)
FALLBACK_WEIGHTS_PATHS = [
    DEFAULT_WEIGHTS_PATH,
    os.path.join(BASE_DIR, "runs", "detect", "sedic_vessel_detector_data2_50ep_b16_n", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "detect", "sedic_vessel_detector_50ep_b16_n", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "detect", "sedic_vessel_detector_150ep_b16", "weights", "best.pt"),
    os.path.join(BASE_DIR, "runs", "detect", "sedic_vessel_detector_fast", "weights", "best.pt"),
    os.path.join(BASE_DIR, "yolov8n.pt"),
    os.path.join(BASE_DIR, "yolov8s.pt"),
]

def get_active_model_path():
    for path in FALLBACK_WEIGHTS_PATHS:
        if os.path.exists(path):
            return path
    return "yolov8n.pt"

# Classes loader
CLASSES_FILE = os.path.join(BASE_DIR, "classes.txt")

DEFAULT_CLASSES = [
    "container_ship",
    "tanker",
    "cargo_ship",
    "passenger_ferry",
    "yacht",
    "speedboat",
    "fishing_boat",
    "military_vessel_local",
    "military_vessel_foreign",
    "military_vessel_general",
    "military_helicopter",
    "military_airfighter",
    "military_submarine",
    "tug_boat"
]

def load_classes():
    if os.path.exists(CLASSES_FILE):
        try:
            with open(CLASSES_FILE, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
                if classes:
                    return classes
        except Exception:
            pass
    return DEFAULT_CLASSES

CLASSES = load_classes()

# Tactical Color Palette for Classes (BGR for OpenCV & HEX for UI)
COLOR_PALETTE_HEX = [
    "#00ffcc",  # 0: container_ship (Cyan Neon)
    "#3b82f6",  # 1: tanker (Marine Blue)
    "#f59e0b",  # 2: cargo_ship (Amber Gold)
    "#ec4899",  # 3: passenger_ferry (Magenta Rose)
    "#eab308",  # 4: yacht (Tactical Yellow)
    "#06b6d4",  # 5: speedboat (Electric Sky)
    "#f97316",  # 6: fishing_boat (Bright Orange)
    "#8b5cf6",  # 7: military_vessel_local (Purple / Navy Local)
    "#ef4444",  # 8: military_vessel_foreign (Alert Red)
    "#64748b",  # 9: military_vessel_general (Slate Military)
    "#10b981",  # 10: military_helicopter (Emerald Radar)
    "#14b8a6",  # 11: military_airfighter (Teal Interceptor)
    "#d946ef",  # 12: military_submarine (Deep Submarine Violet)
    "#84cc16"   # 13: tug_boat (Bright Tactical Lime)
]

def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

CLASS_COLORS_BGR = [hex_to_bgr(c) for c in COLOR_PALETTE_HEX]
CLASS_COLORS_HEX = COLOR_PALETTE_HEX

def get_class_color_bgr(class_id: int):
    return CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]

def get_class_color_hex(class_id: int):
    return CLASS_COLORS_HEX[class_id % len(CLASS_COLORS_HEX)]

# Detection & Tracking Defaults
DEFAULT_CONFIDENCE = 0.50
DEFAULT_IOU = 0.45
DEFAULT_TRACKER = "botsort.yaml"

# Tactical HUD Dark Theme QSS Stylesheet
TACTICAL_DARK_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #090d16;
    color: #e2e8f0;
    font-size: 10pt;
}

QWidget {
    color: #e2e8f0;
    font-size: 10pt;
}

QFrame#HeaderPanel {
    background-color: #0f172a;
    border-bottom: 2px solid #1e293b;
    padding: 8px 16px;
}

QFrame#ControlDeck {
    background-color: #0f172a;
    border-top: 1px solid #1e293b;
    border-radius: 8px;
}

QFrame#SidebarPanel {
    background-color: #0b1120;
    border-left: 1px solid #1e293b;
}

QLabel {
    color: #cbd5e1;
}

QLabel#HeaderTitle {
    color: #f8fafc;
    font-size: 16px;
    font-weight: bold;
    letter-spacing: 1px;
}

QLabel#BadgeStatus {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: bold;
}

QLabel#StatValue {
    font-size: 15px;
    font-weight: bold;
    color: #38bdf8;
}

QLabel#StatLabel {
    font-size: 10px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QPushButton {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #38bdf8;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #0284c7;
    border-color: #0284c7;
}

QPushButton#PrimaryBtn {
    background-color: #0284c7;
    border: 1px solid #38bdf8;
    color: #ffffff;
}

QPushButton#PrimaryBtn:hover {
    background-color: #0369a1;
    border-color: #7dd3fc;
}

QPushButton#DangerBtn {
    background-color: #991b1b;
    border: 1px solid #ef4444;
    color: #fee2e2;
}

QPushButton#DangerBtn:hover {
    background-color: #b91c1c;
    border-color: #f87171;
}

QPushButton#SuccessBtn {
    background-color: #065f46;
    border: 1px solid #10b981;
    color: #d1fae5;
}

QPushButton#SuccessBtn:hover {
    background-color: #047857;
    border-color: #34d399;
}

QComboBox {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    min-width: 90px;
}

QComboBox:hover {
    border-color: #38bdf8;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #0f172a;
    color: #f1f5f9;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
    border: 1px solid #334155;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #0284c7;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #38bdf8;
    border: 2px solid #0f172a;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #7dd3fc;
    border-color: #38bdf8;
}

QScrollArea {
    background-color: transparent;
    border: none;
}

QScrollBar:vertical {
    background: #090d16;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 4px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QProgressBar {
    background-color: #1e293b;
    border-radius: 3px;
    text-align: center;
    font-size: 10px;
    color: #ffffff;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #0284c7;
    border-radius: 3px;
}

QToolTip {
    background-color: #0f172a;
    color: #38bdf8;
    border: 1px solid #38bdf8;
    padding: 4px;
    border-radius: 4px;
}
"""
