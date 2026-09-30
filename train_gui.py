import sys
import os
import re
import subprocess
import torch
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSpinBox,
    QTextEdit, QProgressBar, QFrame, QFileDialog, QMessageBox,
    QGroupBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess
from PyQt6.QtGui import QFont, QIcon, QTextCursor, QColor, QPixmap

from config import TACTICAL_DARK_STYLESHEET, BASE_DIR

class TrainingProcessThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int) # current_epoch, total_epochs
    finished_signal = pyqtSignal(int)      # returncode

    def __init__(self, cmd, cwd):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.process = None
        self._is_killed = False

    def run(self):
        try:
            # Run with unbuffered output and UTF-8 encoding
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )

            total_epochs = 50
            # Check cmd for epochs
            for i, arg in enumerate(self.cmd):
                if arg == "--epochs" and i + 1 < len(self.cmd):
                    try:
                        total_epochs = int(self.cmd[i+1])
                    except ValueError:
                        pass

            epoch_regex = re.compile(r"^\s*(\d+)/(\d+)\s+")

            for line in iter(self.process.stdout.readline, ""):
                if self._is_killed:
                    break
                self.log_signal.emit(line)

                # Parse epoch progress
                match = epoch_regex.search(line.strip())
                if match:
                    cur_ep = int(match.group(1))
                    tot_ep = int(match.group(2))
                    self.progress_signal.emit(cur_ep, tot_ep)

            self.process.stdout.close()
            return_code = self.process.wait()
            self.finished_signal.emit(return_code if not self._is_killed else -1)

        except Exception as e:
            self.log_signal.emit(f"\n[ERROR] Thread execution failed: {str(e)}\n")
            self.finished_signal.emit(-1)

    def stop(self):
        self._is_killed = True
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.kill()
            except Exception:
                pass


class TrainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEDIC 2026 - FORTRESS AI :: AI Training Studio")
        self.resize(1000, 750)
        self.setMinimumSize(850, 600)
        self.setStyleSheet(TACTICAL_DARK_STYLESHEET)

        # Set Window Icon
        logo_path = os.path.join(BASE_DIR, "assets", "fortress_logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))

        self.train_thread = None
        self.init_ui()
        self.check_hardware()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # ---------------------------------------------------------
        # 1. HEADER PANEL
        # ---------------------------------------------------------
        header = QFrame()
        header.setObjectName("HeaderPanel")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        # Top-Left Logo & Title
        title_box = QHBoxLayout()
        title_box.setSpacing(10)
        title_box.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        logo_label = QLabel()
        logo_path = os.path.join(BASE_DIR, "assets", "fortress_logo.png")
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            if not pix.isNull():
                scaled_pix = pix.scaledToHeight(36, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled_pix)
                logo_label.setStyleSheet("background-color: #f8fafc; border-radius: 6px; padding: 2px;")

        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        title = QLabel("SEDIC 2026 - FORTRESS AI")
        title.setObjectName("HeaderTitle")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f8fafc; letter-spacing: 0.5px;")
        subtitle = QLabel("Dedicated YOLO Model Training Studio | UniMAP")
        subtitle.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 500;")
        text_col.addWidget(title)
        text_col.addWidget(subtitle)

        if logo_label.pixmap() is not None:
            title_box.addWidget(logo_label)
        title_box.addLayout(text_col)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        self.lbl_status = QLabel("STATUS: IDLE")
        self.lbl_status.setObjectName("BadgeStatus")
        header_layout.addWidget(self.lbl_status)

        main_layout.addWidget(header)

        # ---------------------------------------------------------
        # 2. CONFIGURATION & PARAMETERS PANEL
        # ---------------------------------------------------------
        config_group = QGroupBox("⚙️ Training Parameters & Hardware Setup")
        config_group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #38bdf8;
                border: 1px solid #1e293b;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
        """)
        config_layout = QGridLayout(config_group)
        config_layout.setContentsMargins(16, 16, 16, 16)
        config_layout.setHorizontalSpacing(20)
        config_layout.setVerticalSpacing(12)

        # Dataset Choice
        lbl_dataset = QLabel("Target Dataset:")
        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems([
            "Videos_Data2 (9 Categories - 15,196 Images)",
            "All Videos_Dataset (Master)"
        ])
        config_layout.addWidget(lbl_dataset, 0, 0)
        config_layout.addWidget(self.combo_dataset, 0, 1, 1, 3)

        # Model Architecture
        lbl_model = QLabel("YOLO Model Architecture:")
        self.combo_model = QComboBox()
        self.combo_model.addItem("YOLOv8n (Nano - Recommended for GTX 1650 Ti)", "n")
        self.combo_model.addItem("YOLOv8s (Small - Balanced)", "s")
        self.combo_model.addItem("YOLOv8m (Medium - High Capacity)", "m")
        config_layout.addWidget(lbl_model, 1, 0)
        config_layout.addWidget(self.combo_model, 1, 1)

        # Epochs
        lbl_epochs = QLabel("Epochs:")
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(150)
        self.spin_epochs.setStyleSheet("padding: 5px; background-color: #1e293b; color: white; border-radius: 4px;")
        config_layout.addWidget(lbl_epochs, 1, 2)
        config_layout.addWidget(self.spin_epochs, 1, 3)

        # Batch Size
        lbl_batch = QLabel("Batch Size:")
        self.combo_batch = QComboBox()
        self.combo_batch.addItems([
            "5 (Pilihan Anda - Batch 5)",
            "16 (Recommended for 4GB VRAM)",
            "8 (Safe Low VRAM)",
            "4 (Ultra-Low VRAM)",
            "32 (Fast High VRAM)"
        ])
        config_layout.addWidget(lbl_batch, 2, 0)
        config_layout.addWidget(self.combo_batch, 2, 1)

        # Image Size
        lbl_imgsz = QLabel("Image Resolution (px):")
        self.combo_imgsz = QComboBox()
        self.combo_imgsz.addItems(["512 (Recommended)", "480", "416", "640"])
        config_layout.addWidget(lbl_imgsz, 2, 2)
        config_layout.addWidget(self.combo_imgsz, 2, 3)

        # Hardware Info Label
        self.lbl_hw_info = QLabel("Hardware: Checking...")
        self.lbl_hw_info.setStyleSheet("color: #34d399; font-weight: bold; font-size: 11px;")
        config_layout.addWidget(self.lbl_hw_info, 3, 0, 1, 4)

        main_layout.addWidget(config_group)

        # ---------------------------------------------------------
        # 3. ACTION CONTROLS DECK
        # ---------------------------------------------------------
        controls_frame = QFrame()
        controls_frame.setObjectName("ControlDeck")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setSpacing(12)

        self.btn_train = QPushButton("⚡ MULA TRAINING (DALAM GUI)")
        self.btn_train.setObjectName("SuccessBtn")
        self.btn_train.setFixedHeight(40)
        self.btn_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_train.clicked.connect(self.start_training)
        controls_layout.addWidget(self.btn_train)

        self.btn_terminal = QPushButton("🖥️ Buka Terminal Luaran (CMD)")
        self.btn_terminal.setObjectName("PrimaryBtn")
        self.btn_terminal.setFixedHeight(40)
        self.btn_terminal.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_terminal.clicked.connect(self.launch_external_terminal_train)
        controls_layout.addWidget(self.btn_terminal)

        self.btn_stop = QPushButton("⏹ HENTIKAN")
        self.btn_stop.setObjectName("DangerBtn")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_training)
        controls_layout.addWidget(self.btn_stop)

        self.btn_open_runs = QPushButton("📁 Folder Result")
        self.btn_open_runs.setFixedHeight(40)
        self.btn_open_runs.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_runs.clicked.connect(self.open_runs_folder)
        controls_layout.addWidget(self.btn_open_runs)

        self.btn_launch_gui = QPushButton("🚀 Surveillance HUD")
        self.btn_launch_gui.setFixedHeight(40)
        self.btn_launch_gui.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_launch_gui.clicked.connect(self.launch_surveillance_hud)
        controls_layout.addWidget(self.btn_launch_gui)

        main_layout.addWidget(controls_frame)

        # ---------------------------------------------------------
        # 4. PROGRESS BAR & METRICS
        # ---------------------------------------------------------
        progress_box = QVBoxLayout()
        progress_hdr = QHBoxLayout()
        
        self.lbl_progress_text = QLabel("Training Progress: 0 / 0 Epochs (0%)")
        self.lbl_progress_text.setStyleSheet("font-weight: bold; color: #38bdf8;")
        progress_hdr.addWidget(self.lbl_progress_text)
        
        progress_hdr.addStretch()
        progress_box.addLayout(progress_hdr)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        progress_box.addWidget(self.progress_bar)

        main_layout.addLayout(progress_box)

        # ---------------------------------------------------------
        # 5. LIVE TERMINAL LOG CONSOLE
        # ---------------------------------------------------------
        lbl_console = QLabel("📋 Live Terminal Output & Training Logs:")
        lbl_console.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        main_layout.addWidget(lbl_console)

        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setFont(QFont("Consolas", 10))
        self.txt_console.setStyleSheet("""
            QTextEdit {
                background-color: #060911;
                color: #e2e8f0;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        main_layout.addWidget(self.txt_console, stretch=1)

    def check_hardware(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.lbl_hw_info.setText(f"🚀 [HARDWARE] NVIDIA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) - CUDA Acceleration Ready")
            self.lbl_hw_info.setStyleSheet("color: #34d399; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_hw_info.setText("⚠️ [HARDWARE] No CUDA GPU Detected. Training will utilize CPU (Slower).")
            self.lbl_hw_info.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 11px;")

    def log(self, text: str):
        self.txt_console.moveCursor(QTextCursor.MoveOperation.End)
        self.txt_console.insertPlainText(text)
        self.txt_console.moveCursor(QTextCursor.MoveOperation.End)

    def launch_external_terminal_train(self):
        is_data2_only = (self.combo_dataset.currentIndex() == 0)
        script_name = "train_data2.py" if is_data2_only else "train.py"
        model_code = self.combo_model.currentData() or "n"
        epochs = self.spin_epochs.value()
        batch_text = self.combo_batch.currentText()
        batch_size = int(batch_text.split()[0])
        imgsz = int(self.combo_imgsz.currentText().split()[0])

        win_title = "SEDIC 2026 - FORTRESS AI Training Terminal"
        cmd_to_run = f"python {script_name} --epochs {epochs} --batch {batch_size} --model {model_code} --imgsz {imgsz}"
        cmd_str = f'start "{win_title}" /D "{BASE_DIR}" cmd.exe /k "{cmd_to_run}"'
        subprocess.Popen(cmd_str, shell=True)


    def start_training(self):
        # Determine script
        is_data2_only = (self.combo_dataset.currentIndex() == 0)
        script_name = "train_data2.py" if is_data2_only else "train.py"
        script_path = os.path.join(BASE_DIR, script_name)

        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Error", f"Script tidak ditemui: {script_path}")
            return

        model_code = self.combo_model.currentData() or "n"
        epochs = self.spin_epochs.value()
        
        # Batch size
        batch_text = self.combo_batch.currentText()
        batch_size = int(batch_text.split()[0])

        # Imgsz
        imgsz = int(self.combo_imgsz.currentText().split()[0])

        cmd = [
            sys.executable,
            "-u", # unbuffered binary stdout and stderr
            script_path,
            "--epochs", str(epochs),
            "--batch", str(batch_size),
            "--model", str(model_code),
            "--imgsz", str(imgsz)
        ]

        self.txt_console.clear()
        self.log(f"===============================================================\n")
        self.log(f"🚀 MEMULAKAN PROSES TRAINING YOLOv8 [{model_code.upper()}]\n")
        self.log(f"📁 Target Dataset : {'Videos_Data2 Sahaja' if is_data2_only else 'Semua Videos_Dataset'}\n")
        self.log(f"⚙️ Hyperparameters: Epochs={epochs}, Batch={batch_size}, ImgSz={imgsz}px\n")
        self.log(f"⚡ Command        : {' '.join(cmd)}\n")
        self.log(f"===============================================================\n\n")

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("STATUS: TRAINING...")
        self.lbl_status.setStyleSheet("background-color: #0284c7; color: white; border: 1px solid #38bdf8; border-radius: 4px; padding: 3px 8px;")
        self.progress_bar.setValue(0)
        self.lbl_progress_text.setText(f"Training Progress: 0 / {epochs} Epochs (0%)")

        self.train_thread = TrainingProcessThread(cmd, cwd=BASE_DIR)
        self.train_thread.log_signal.connect(self.log)
        self.train_thread.progress_signal.connect(self.on_progress_update)
        self.train_thread.finished_signal.connect(self.on_training_finished)
        self.train_thread.start()

    def on_progress_update(self, current_epoch, total_epochs):
        percent = int((current_epoch / max(total_epochs, 1)) * 100)
        self.progress_bar.setValue(percent)
        self.lbl_progress_text.setText(f"Training Progress: Epoch {current_epoch} / {total_epochs} ({percent}%)")

    def stop_training(self):
        reply = QMessageBox.question(
            self, "Hentikan Training",
            "Adakah anda pasti mahu menghentikan proses training sekarang?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.train_thread and self.train_thread.isRunning():
                self.train_thread.stop()
                self.log("\n⚠️ [INFO] Training dihentikan oleh pengguna.\n")
                self.btn_train.setEnabled(True)
                self.btn_stop.setEnabled(False)
                self.lbl_status.setText("STATUS: STOPPED")
                self.lbl_status.setStyleSheet("background-color: #991b1b; color: white; border: 1px solid #ef4444; border-radius: 4px; padding: 3px 8px;")

    def on_training_finished(self, return_code):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if return_code == 0:
            self.progress_bar.setValue(100)
            self.lbl_status.setText("STATUS: COMPLETED")
            self.lbl_status.setStyleSheet("background-color: #065f46; color: white; border: 1px solid #10b981; border-radius: 4px; padding: 3px 8px;")
            self.log("\n\n🎉 [TAHNIAH] Training selesai dengan jayanya! Weights baru disimpan di folder 'runs/detect/'.\n")
            QMessageBox.information(self, "Training Selesai", "Training telah selesai dengan jayanya!\nWeights terbaik disimpan di runs/detect/.")
        else:
            self.lbl_status.setText("STATUS: ERROR / STOPPED")
            self.lbl_status.setStyleSheet("background-color: #991b1b; color: white; border: 1px solid #ef4444; border-radius: 4px; padding: 3px 8px;")

    def open_runs_folder(self):
        runs_dir = os.path.join(BASE_DIR, "runs", "detect")
        os.makedirs(runs_dir, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(runs_dir)
        else:
            subprocess.Popen(["xdg-open", runs_dir])

    def launch_surveillance_hud(self):
        main_py = os.path.join(BASE_DIR, "main.py")
        if os.path.exists(main_py):
            subprocess.Popen([sys.executable, main_py], cwd=BASE_DIR)
        else:
            QMessageBox.warning(self, "Warning", "main.py tidak ditemui.")


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("SEDIC 2026 - FORTRESS AI Training Studio")
    gui = TrainGUI()
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
