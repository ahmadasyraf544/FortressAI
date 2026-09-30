import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from PyQt6.QtGui import QIcon, QFont
from config import BASE_DIR
from ui_main import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("SEDIC 2026 - FORTRESS AI")
    app.setOrganizationName("SEDIC 2026")

    logo_path = os.path.join(BASE_DIR, "assets", "fortress_logo.png")
    if os.path.exists(logo_path):
        app.setWindowIcon(QIcon(logo_path))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
