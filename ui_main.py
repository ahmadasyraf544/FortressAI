import os
import json
import csv
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QComboBox, QHBoxLayout, QVBoxLayout, QGridLayout, QFrame,
    QScrollArea, QFileDialog, QMessageBox, QCheckBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QFont, QIcon
from PyQt6.QtCore import Qt, pyqtSlot

from config import (
    CLASSES, DEFAULT_CONFIDENCE, DEFAULT_IOU,
    TACTICAL_DARK_STYLESHEET, get_active_model_path, get_class_color_hex, BASE_DIR
)
from video_thread import VideoProcessingThread
from hud_card import TargetHUDCard, TargetDetailModal

class VideoViewerWidget(QLabel):
    """Custom QLabel that maintains video aspect ratio cleanly and displays a tactical standby screen."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 420)
        self.setStyleSheet("background-color: #060911; border: 1px solid #1e293b; border-radius: 8px;")
        self.current_pixmap = None
        self.show_standby()

    def show_standby(self):
        self.current_pixmap = None
        self.update()

    def set_frame(self, q_img: QImage):
        self.current_pixmap = QPixmap.fromImage(q_img)
        self.update_display()

    def update_display(self):
        if self.current_pixmap:
            scaled_pix = self.current_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pix)
        else:
            self.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_pixmap is None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            w, h = self.width(), self.height()
            cx, cy = w // 2, h // 2

            # Tactical Radar Grid
            pen_grid = QPen(QColor("#1e293b"), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen_grid)
            painter.drawLine(cx, 40, cx, h - 40)
            painter.drawLine(40, cy, w - 40, cy)
            
            pen_circle = QPen(QColor("#0f172a"), 2)
            painter.setPen(pen_circle)
            painter.drawEllipse(cx - 120, cy - 120, 240, 240)
            painter.drawEllipse(cx - 60, cy - 60, 120, 120)

            # Standby Text
            painter.setPen(QColor("#38bdf8"))
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "STANDBY: SELECT WEBCAM OR LOAD VIDEO FILE")
            
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(cx - 160, cy + 32, 320, 20, Qt.AlignmentFlag.AlignCenter, "SEDIC 2026 - FORTRESS AI :: MARITIME SURVEILLANCE")


class MainWindow(QMainWindow):
    """
    Main Application Window: Dual-Input Viewer + BoT-SORT Tracking + Vertical Analytics HUD.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEDIC 2026 - FORTRESS AI :: Tactical Maritime Surveillance HUD")
        self.resize(1340, 820)
        self.setMinimumSize(1080, 680)

        # Set Window Icon
        logo_path = os.path.join(BASE_DIR, "assets", "fortress_logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))

        # Apply Global Tactical Dark Theme
        self.setStyleSheet(TACTICAL_DARK_STYLESHEET)

        # Data stores
        self.logged_targets = []  # List of target_data dicts
        self.active_filter = "ALL"

        # Initialize Background Video Thread
        self.video_thread = VideoProcessingThread(model_path=get_active_model_path())
        self.video_thread.frame_ready.connect(self.on_frame_ready)
        self.video_thread.new_target_detected.connect(self.on_new_target_detected)
        self.video_thread.status_changed.connect(self.on_status_changed)
        self.video_thread.error_occurred.connect(self.on_error_occurred)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 8, 10, 10)
        main_layout.setSpacing(8)

        # 1. TOP HEADER & METRICS BAR
        main_layout.addWidget(self.create_header_bar())

        # 2. BODY SPLIT: Left Viewport (70%) + Right Analytics HUD (30%)
        body_layout = QHBoxLayout()
        body_layout.setSpacing(10)

        # Left Column: Video Viewer & Control Deck
        left_column = QVBoxLayout()
        left_column.setSpacing(8)

        self.viewer = VideoViewerWidget()
        left_column.addWidget(self.viewer, stretch=1)

        # Video progress slider (for video playback)
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)
        self.progress_slider.sliderMoved.connect(self.on_seek_slider)
        left_column.addWidget(self.progress_slider)

        # Control Deck
        left_column.addWidget(self.create_control_deck())

        body_layout.addLayout(left_column, stretch=7)

        # Right Column: Analytics HUD Sidebar
        body_layout.addWidget(self.create_sidebar_hud(), stretch=3)

        main_layout.addLayout(body_layout)

    def create_header_bar(self):
        header_frame = QFrame()
        header_frame.setObjectName("HeaderPanel")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 6, 12, 6)
        header_layout.setSpacing(16)

        # Top-Left: Logo & App Title Container
        title_box = QHBoxLayout()
        title_box.setSpacing(10)
        title_box.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Fortress Logo Image
        logo_label = QLabel()
        logo_path = os.path.join(BASE_DIR, "assets", "fortress_logo.png")
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            if not pix.isNull():
                scaled_pix = pix.scaledToHeight(38, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled_pix)
                logo_label.setStyleSheet("background-color: #f8fafc; border-radius: 6px; padding: 2px;")

        if logo_label.pixmap() is None or logo_label.pixmap().isNull():
            logo_label.setText("🛡️")
            logo_label.setStyleSheet("font-size: 20px;")

        # Title + Subtitle Column
        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        text_col.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        title_text = QLabel("SEDIC 2026 - FORTRESS AI")
        title_text.setObjectName("HeaderTitle")
        title_text.setStyleSheet("font-size: 15px; font-weight: bold; color: #f8fafc; letter-spacing: 0.5px;")

        subtitle_text = QLabel("Tactical Maritime Surveillance & Analytics HUD | UniMAP")
        subtitle_text.setStyleSheet("font-size: 10px; color: #94a3b8; font-weight: 500;")

        text_col.addWidget(title_text)
        text_col.addWidget(subtitle_text)

        title_box.addWidget(logo_label)
        title_box.addLayout(text_col)
        header_layout.addLayout(title_box)

        # Live Status Tag
        self.status_badge = QLabel("● STANDBY")
        self.status_badge.setObjectName("BadgeStatus")
        header_layout.addWidget(self.status_badge)

        header_layout.addStretch()

        # Metrics Badges
        def make_metric(label_str):
            box = QVBoxLayout()
            box.setSpacing(1)
            box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("0")
            val.setObjectName("StatValue")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl = QLabel(label_str)
            lbl.setObjectName("StatLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(val)
            box.addWidget(lbl)
            return box, val

        box_fps, self.val_fps = make_metric("FPS")
        box_tracks, self.val_tracks = make_metric("ACTIVE TRACKS")
        box_unique, self.val_unique = make_metric("TOTAL TARGETS")
        box_dets, self.val_dets = make_metric("DETECTIONS")

        header_layout.addLayout(box_fps)
        header_layout.addLayout(box_tracks)
        header_layout.addLayout(box_unique)
        header_layout.addLayout(box_dets)

        return header_frame

    def create_control_deck(self):
        deck = QFrame()
        deck.setObjectName("ControlDeck")
        layout = QVBoxLayout(deck)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Top Row: Buttons & Inputs
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        # Webcam group
        self.webcam_combo = QComboBox()
        self.webcam_combo.addItems(["Webcam #0", "Webcam #1", "Webcam #2", "Webcam #3"])
        self.webcam_btn = QPushButton("📷 Open Web Camera")
        self.webcam_btn.setObjectName("PrimaryBtn")
        self.webcam_btn.clicked.connect(self.on_open_webcam)

        btn_row.addWidget(self.webcam_combo)
        btn_row.addWidget(self.webcam_btn)

        # Video File
        self.file_btn = QPushButton("📁 Load Video File")
        self.file_btn.clicked.connect(self.on_load_video)
        btn_row.addWidget(self.file_btn)

        # AI Training Studio
        self.train_btn = QPushButton("⚡ AI Training Studio")
        self.train_btn.setObjectName("SuccessBtn")
        self.train_btn.setToolTip("Buka GUI Training untuk Videos_Data2")
        self.train_btn.clicked.connect(self.on_open_train_studio)
        btn_row.addWidget(self.train_btn)

        # Pause / Stop
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.on_toggle_pause)
        self.pause_btn.setEnabled(False)
        btn_row.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("DangerBtn")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)

        layout.addLayout(btn_row)

        # Bottom Row: Sliders & Thresholds
        slider_row = QHBoxLayout()
        slider_row.setSpacing(14)

        # Confidence Slider
        slider_row.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(int(DEFAULT_CONFIDENCE * 100))
        self.conf_val_label = QLabel(f"{int(DEFAULT_CONFIDENCE * 100)}%")
        self.conf_val_label.setStyleSheet("color: #38bdf8; font-weight: bold; min-width: 38px;")
        self.conf_slider.valueChanged.connect(self.on_conf_changed)

        slider_row.addWidget(self.conf_slider)
        slider_row.addWidget(self.conf_val_label)

        # IoU Slider
        slider_row.addWidget(QLabel("IoU:"))
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(10, 100)
        self.iou_slider.setValue(int(DEFAULT_IOU * 100))
        self.iou_val_label = QLabel(f"{DEFAULT_IOU:.2f}")
        self.iou_val_label.setStyleSheet("color: #38bdf8; font-weight: bold; min-width: 32px;")
        self.iou_slider.valueChanged.connect(self.on_iou_changed)

        slider_row.addWidget(self.iou_slider)
        slider_row.addWidget(self.iou_val_label)

        # Resolution / Speed Controls
        slider_row.addWidget(QLabel("Size:"))
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItem("384px (Max FPS)", 384)
        self.imgsz_combo.addItem("480px (Balanced)", 480)
        self.imgsz_combo.addItem("640px (High Res)", 640)
        self.imgsz_combo.setCurrentIndex(1)  # Default 480px
        self.imgsz_combo.currentIndexChanged.connect(self.on_imgsz_changed)
        slider_row.addWidget(self.imgsz_combo)

        self.max_speed_cb = QCheckBox("Max GPU Speed")
        self.max_speed_cb.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        self.max_speed_cb.toggled.connect(self.on_speed_mode_toggled)
        slider_row.addWidget(self.max_speed_cb)

        # Tracker Tag
        tracker_tag = QLabel("Tracker: BoT-SORT")
        tracker_tag.setStyleSheet("color: #94a3b8; font-size: 11px; padding-left: 6px;")
        slider_row.addWidget(tracker_tag)

        layout.addLayout(slider_row)

        return deck

    def create_sidebar_hud(self):
        sidebar = QFrame()
        sidebar.setObjectName("SidebarPanel")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(8)

        # Sidebar Header
        sb_header = QHBoxLayout()
        sb_title = QLabel("🎯 ANALYTICS HUD")
        sb_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8; letter-spacing: 0.5px;")
        
        self.target_count_badge = QLabel("0 Targets")
        self.target_count_badge.setStyleSheet("""
            background-color: #1e293b; color: #94a3b8; border-radius: 4px;
            padding: 2px 6px; font-size: 11px; font-weight: bold;
        """)
        sb_header.addWidget(sb_title)
        sb_header.addStretch()
        sb_header.addWidget(self.target_count_badge)
        sidebar_layout.addLayout(sb_header)

        # Filter dropdown by class
        filter_box = QHBoxLayout()
        filter_box.setSpacing(6)
        filter_lbl = QLabel("Filter:")
        filter_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("ALL CLASSES", "ALL")
        for cls_name in CLASSES:
            self.filter_combo.addItem(cls_name.upper().replace("_", " "), cls_name)
        self.filter_combo.currentIndexChanged.connect(self.on_filter_changed)

        filter_box.addWidget(filter_lbl)
        filter_box.addWidget(self.filter_combo, stretch=1)
        sidebar_layout.addLayout(filter_box)

        # Scroll Area for Target Cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.card_container = QWidget()
        self.card_layout = QVBoxLayout(self.card_container)
        self.card_layout.setContentsMargins(0, 0, 0, 0)
        self.card_layout.setSpacing(6)
        self.card_layout.addStretch()  # Keep cards top-aligned

        self.scroll_area.setWidget(self.card_container)
        sidebar_layout.addWidget(self.scroll_area, stretch=1)

        # Bottom Tools: Auto scroll & Actions
        bottom_box = QHBoxLayout()
        self.auto_scroll_cb = QCheckBox("Auto Scroll")
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.setStyleSheet("color: #94a3b8; font-size: 11px;")

        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        clear_btn.clicked.connect(self.on_clear_cards)

        export_btn = QPushButton("Export Log")
        export_btn.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        export_btn.clicked.connect(self.on_export_log)

        bottom_box.addWidget(self.auto_scroll_cb)
        bottom_box.addStretch()
        bottom_box.addWidget(clear_btn)
        bottom_box.addWidget(export_btn)
        sidebar_layout.addLayout(bottom_box)

        return sidebar

    # ==========================================================================
    # SLOTS & CONTROLLER LOGIC
    # ==========================================================================

    @pyqtSlot()
    def on_open_webcam(self):
        cam_idx = self.webcam_combo.currentIndex()
        self.video_thread.stop()
        self.video_thread.set_source("webcam", cam_idx)
        self.video_thread.start()
        
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ Pause")
        self.stop_btn.setEnabled(True)
        self.status_badge.setText("● LIVE WEBCAM")
        self.status_badge.setStyleSheet("background-color: #064e3b; color: #34d399; border: 1px solid #059669;")

    @pyqtSlot()
    def on_load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if file_path:
            self.video_thread.stop()
            self.video_thread.set_source("video", file_path)
            self.video_thread.start()

            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("⏸ Pause")
            self.stop_btn.setEnabled(True)
            self.status_badge.setText("● PLAYING VIDEO")
            self.status_badge.setStyleSheet("background-color: #1e3a8a; color: #60a5fa; border: 1px solid #3b82f6;")

    @pyqtSlot()
    def on_toggle_pause(self):
        self.video_thread.toggle_pause()
        if self.video_thread.is_paused:
            self.pause_btn.setText("▶ Resume")
            self.status_badge.setText("● PAUSED")
            self.status_badge.setStyleSheet("background-color: #78350f; color: #fbbf24; border: 1px solid #d97706;")
        else:
            self.pause_btn.setText("⏸ Pause")
            self.status_badge.setText("● LIVE FEED")
            self.status_badge.setStyleSheet("background-color: #064e3b; color: #34d399; border: 1px solid #059669;")

    @pyqtSlot()
    def on_stop(self):
        self.video_thread.stop()
        self.viewer.show_standby()
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_badge.setText("● STANDBY")
        self.status_badge.setStyleSheet("background-color: #1e293b; color: #94a3b8; border: 1px solid #334155;")
        self.val_fps.setText("0.0")
        self.val_tracks.setText("0")
        self.val_dets.setText("0")
        self.progress_slider.setValue(0)

    @pyqtSlot()
    def on_open_train_studio(self):
        import subprocess, sys
        from config import BASE_DIR
        train_gui_script = os.path.join(BASE_DIR, "train_gui.py")
        if os.path.exists(train_gui_script):
            subprocess.Popen([sys.executable, train_gui_script], cwd=BASE_DIR)
        else:
            QMessageBox.warning(self, "Error", "train_gui.py tidak ditemui.")

    @pyqtSlot(int)
    def on_conf_changed(self, val: int):
        conf_float = val / 100.0
        self.conf_val_label.setText(f"{val}%")
        self.video_thread.update_confidence(conf_float)

    @pyqtSlot(int)
    def on_iou_changed(self, val: int):
        iou_float = val / 100.0
        self.iou_val_label.setText(f"{iou_float:.2f}")
        self.video_thread.update_iou(iou_float)

    @pyqtSlot(int)
    def on_seek_slider(self, val: int):
        percent = val / 10.0
        self.video_thread.seek_percentage(percent)

    @pyqtSlot(int)
    def on_imgsz_changed(self, index: int):
        imgsz = self.imgsz_combo.currentData()
        if imgsz:
            self.video_thread.set_imgsz(imgsz)

    @pyqtSlot(bool)
    def on_speed_mode_toggled(self, checked: bool):
        self.video_thread.set_speed_mode(checked)

    @pyqtSlot(QImage, dict)
    def on_frame_ready(self, q_img: QImage, metrics: dict):
        self.viewer.set_frame(q_img)
        self.val_fps.setText(f"{metrics.get('fps', 0.0):.1f}")
        self.val_tracks.setText(str(metrics.get("active_tracks", 0)))
        self.val_unique.setText(str(metrics.get("total_unique", len(self.logged_targets))))
        self.val_dets.setText(str(metrics.get("detections_current", 0)))

        # Update progress slider if playing video
        if self.video_thread.source_type == "video" and not self.progress_slider.isSliderDown():
            prog = metrics.get("progress_percent", 0.0)
            self.progress_slider.setValue(int(prog * 10))

    @pyqtSlot(dict)
    def on_new_target_detected(self, target_data: dict):
        """Event-driven: creates snapshot HUD card whenever a new object ID is first spotted!"""
        self.logged_targets.append(target_data)
        self.target_count_badge.setText(f"{len(self.logged_targets)} Targets")

        card = TargetHUDCard(target_data)
        card.clicked.connect(self.open_target_modal)

        # Check class filter
        if self.active_filter != "ALL" and target_data.get("class_name") != self.active_filter:
            card.setVisible(False)

        # Insert at the top (before the stretch item)
        self.card_layout.insertWidget(0, card)

        if self.auto_scroll_cb.isChecked():
            self.scroll_area.verticalScrollBar().setValue(0)

    @pyqtSlot(int)
    def on_filter_changed(self, index: int):
        selected_class = self.filter_combo.currentData()
        self.active_filter = selected_class or "ALL"

        for i in range(self.card_layout.count()):
            item = self.card_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), TargetHUDCard):
                card = item.widget()
                if self.active_filter == "ALL":
                    card.setVisible(True)
                else:
                    card.setVisible(card.class_name == self.active_filter)

    def open_target_modal(self, target_data: dict):
        modal = TargetDetailModal(target_data, self)
        modal.exec()

    @pyqtSlot()
    def on_clear_cards(self):
        self.logged_targets.clear()
        self.target_count_badge.setText("0 Targets")
        while self.card_layout.count() > 1:
            item = self.card_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    @pyqtSlot()
    def on_export_log(self):
        if not self.logged_targets:
            QMessageBox.information(self, "Export Log", "No targets detected in current session.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Target Log", f"target_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Format (*.json);;CSV Format (*.csv)"
        )
        if not file_path:
            return

        try:
            if file_path.endswith(".csv"):
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Track ID", "Class Name", "Confidence", "Timestamp", "Bounding Box"])
                    for t in self.logged_targets:
                        writer.writerow([t["track_id"], t["class_name"], f"{t['confidence']*100:.2f}%", t["timestamp"], str(t["bbox"])])
            else:
                export_data = []
                for t in self.logged_targets:
                    export_data.append({
                        "track_id": t["track_id"],
                        "class_name": t["class_name"],
                        "confidence": float(f"{t['confidence']:.4f}"),
                        "timestamp": t["timestamp"],
                        "bbox": t["bbox"]
                    })
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=4)

            QMessageBox.information(self, "Export Successful", f"Target log saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save log: {e}")

    @pyqtSlot(str)
    def on_status_changed(self, msg: str):
        self.statusBar().showMessage(msg, 4000)

    @pyqtSlot(str)
    def on_error_occurred(self, err_msg: str):
        self.statusBar().showMessage(f"Error: {err_msg}", 5000)
        QMessageBox.warning(self, "System Warning", err_msg)

    def closeEvent(self, event):
        self.video_thread.stop()
        super().closeEvent(event)
