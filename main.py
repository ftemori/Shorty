import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor

from src.ui.main_window import MainWindow

def main():
    # Force software video decoding to prevent CUDA/hardware acceleration errors
    os.environ['QT_MEDIA_BACKEND'] = 'ffmpeg'
    os.environ['LIBVA_DRIVER_NAME'] = 'i965'  # Force software VA-API if available
    os.environ['VDPAU_DRIVER'] = 'va_gl'  # Fallback VDPAU driver
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

