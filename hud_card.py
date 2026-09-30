import os
import cv2
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, 
    QProgressBar, QDialog, QPushButton, QFileDialog, QGraphicsDropShadowEffect
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QCursor, QFont
from PyQt6.QtCore import Qt, pyqtSignal

from config import get_class_color_hex, get_class_color_bgr

def numpy_to_pixmap(cv_img, target_width=None, target_height=None):
    """Converts an OpenCV BGR numpy image to QPixmap."""
    if cv_img is None or cv_img.size == 0:
        pix = QPixmap(80, 80)
        pix.fill(QColor("#1e293b"))
        return pix
    
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(q_img)
    
    if target_width and target_height:
        pix = pix.scaled(
            target_width, target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    return pix

class TargetHUDCard(QFrame):
    """
    Tactical HUD Card displaying an event-driven snapshot of an identified target.
    Emits `clicked(dict)` on user interaction to trigger modal inspection.
    """
    clicked = pyqtSignal(dict)

    def __init__(self, target_data: dict, parent=None):
        super().__init__(parent)
        self.target_data = target_data
        self.class_id = target_data.get("class_id", 0)
        self.class_name = target_data.get("class_name", "Unknown")
        self.track_id = target_data.get("track_id", 0)
        self.conf = target_data.get("confidence", 0.0)
        self.timestamp = target_data.get("timestamp", datetime.now().strftime("%H:%M:%S"))
        self.crop_img = target_data.get("crop_image", None)
        self.hex_color = get_class_color_hex(self.class_id)

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("TargetHUDCard")
        self.setStyleSheet(f"""
            QFrame#TargetHUDCard {{
                background-color: #131d31;
                border: 1px solid #1e293b;
                border-left: 4px solid {self.hex_color};
                border-radius: 8px;
                margin-bottom: 6px;
            }}
            QFrame#TargetHUDCard:hover {{
                background-color: #1a2744;
                border-color: {self.hex_color};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 10, 8)
        layout.setSpacing(10)

        # 1. Thumbnail Preview Frame
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(72, 72)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet(f"""
            background-color: #090d16;
            border: 1px solid #334155;
            border-radius: 6px;
        """)
        
        pixmap = numpy_to_pixmap(self.crop_img, 70, 70)
        self.thumb_label.setPixmap(pixmap)
        layout.addWidget(self.thumb_label)

        # 2. Right Info Panel
        info_layout = QVBoxLayout()
        info_layout.setSpacing(3)
        info_layout.setContentsMargins(0, 0, 0, 0)

        # Top row: Target ID & Timestamp
        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        
        target_badge = QLabel(f"#{self.track_id:02d}")
        target_badge.setStyleSheet(f"""
            color: #ffffff;
            background-color: #1e293b;
            border: 1px solid #475569;
            border-radius: 3px;
            padding: 1px 5px;
            font-size: 11px;
            font-weight: bold;
        """)
        
        class_badge = QLabel(self.class_name.upper().replace("_", " "))
        class_badge.setStyleSheet(f"""
            color: {self.hex_color};
            font-size: 12px;
            font-weight: bold;
        """)
        class_badge.setWordWrap(True)

        top_row.addWidget(target_badge)
        top_row.addWidget(class_badge)
        top_row.addStretch()
        
        time_label = QLabel(self.timestamp)
        time_label.setStyleSheet("color: #64748b; font-size: 10px;")
        top_row.addWidget(time_label)

        info_layout.addLayout(top_row)

        # Confidence Score Bar
        conf_row = QHBoxLayout()
        conf_row.setSpacing(6)
        
        conf_text = QLabel(f"{self.conf * 100:.1f}%")
        conf_text.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        
        self.conf_bar = QProgressBar()
        self.conf_bar.setFixedHeight(6)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(int(self.conf * 100))
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {self.hex_color};
                border-radius: 2px;
            }}
        """)

        conf_row.addWidget(self.conf_bar)
        conf_row.addWidget(conf_text)
        info_layout.addLayout(conf_row)

        # Bounding box info
        bbox = self.target_data.get("bbox", [0, 0, 0, 0])
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        dim_label = QLabel(f"ROI: {w}x{h} px | Track BoT-SORT")
        dim_label.setStyleSheet("color: #475569; font-size: 9px;")
        info_layout.addWidget(dim_label)

        layout.addLayout(info_layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.target_data)
        super().mousePressEvent(event)


class TargetDetailModal(QDialog):
    """
    Detailed Modal Dialog triggered when clicking a Target HUD card.
    Provides high-res crop inspection, bounding box metrics & export options.
    """
    def __init__(self, target_data: dict, parent=None):
        super().__init__(parent)
        self.target_data = target_data
        self.setWindowTitle(f"TARGET INSPECTION: #{target_data.get('track_id', 0)} - {target_data.get('class_name', '')}")
        self.setMinimumSize(540, 500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        hex_color = get_class_color_hex(self.target_data.get("class_id", 0))

        # Title Header
        header_layout = QHBoxLayout()
        title = QLabel(f"TARGET IDENTIFICATION: #{self.target_data.get('track_id', 0):02d}")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {hex_color}; letter-spacing: 1px;")
        
        status_tag = QLabel("ACTIVE TRACK")
        status_tag.setStyleSheet("""
            background-color: #064e3b; color: #34d399; border: 1px solid #059669;
            border-radius: 4px; padding: 3px 8px; font-size: 11px; font-weight: bold;
        """)
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(status_tag)
        layout.addLayout(header_layout)

        # High-Res Image Display
        img_frame = QFrame()
        img_frame.setStyleSheet(f"background-color: #090d16; border: 2px solid {hex_color}; border-radius: 8px;")
        img_layout = QVBoxLayout(img_frame)
        img_layout.setContentsMargins(8, 8, 8, 8)
        
        crop_img = self.target_data.get("crop_image", None)
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if crop_img is not None and crop_img.size > 0:
            pix = numpy_to_pixmap(crop_img, 480, 280)
            img_label.setPixmap(pix)
        else:
            img_label.setText("No Snapshot Available")
            img_label.setStyleSheet("color: #64748b; font-size: 14px;")

        img_layout.addWidget(img_label)
        layout.addWidget(img_frame)

        # Detail Metrics Grid
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 10px;")
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_layout.setSpacing(6)

        cls_name = self.target_data.get("class_name", "Unknown").upper().replace("_", " ")
        conf = self.target_data.get("confidence", 0.0) * 100
        timestamp = self.target_data.get("timestamp", "N/A")
        bbox = self.target_data.get("bbox", [0, 0, 0, 0])
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])

        def create_metric_row(label, value, val_color="#f8fafc"):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
            val = QLabel(str(value))
            val.setStyleSheet(f"color: {val_color}; font-weight: bold; font-size: 12px;")
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            return row

        metrics_layout.addLayout(create_metric_row("Classification Category:", cls_name, hex_color))
        metrics_layout.addLayout(create_metric_row("Confidence Probability:", f"{conf:.2f}%", "#38bdf8"))
        metrics_layout.addLayout(create_metric_row("Detection Time (First Sighting):", timestamp))
        metrics_layout.addLayout(create_metric_row("Bounding Box Coordinates [X1, Y1, X2, Y2]:", f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"))
        metrics_layout.addLayout(create_metric_row("ROI Dimensions (Width x Height):", f"{w} x {h} pixels"))

        layout.addWidget(metrics_frame)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        save_btn = QPushButton("💾 Save Snapshot PNG")
        save_btn.setObjectName("PrimaryBtn")
        save_btn.clicked.connect(self.save_snapshot)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def save_snapshot(self):
        crop_img = self.target_data.get("crop_image", None)
        if crop_img is None or crop_img.size == 0:
            return

        default_name = f"target_{self.target_data.get('track_id', 0)}_{self.target_data.get('class_name', 'vessel')}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Target Snapshot", default_name, "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if path:
            cv2.imwrite(path, crop_img)
