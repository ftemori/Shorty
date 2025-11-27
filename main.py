import sys
import os
import threading
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QFontDatabase

from src.ui.main_window import MainWindow

class StderrFilter:
    def __init__(self):
        self.original_stderr_fd = os.dup(sys.stderr.fileno())
        self.r_fd, self.w_fd = os.pipe()
        self.stop_event = threading.Event()
        
        # Redirect stderr to pipe
        os.dup2(self.w_fd, sys.stderr.fileno())
        
        # Start reader thread
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()

    def _reader_thread(self):
        with os.fdopen(self.r_fd, 'r', errors='replace') as pipe_reader:
            with os.fdopen(self.original_stderr_fd, 'w') as real_stderr:
                while not self.stop_event.is_set():
                    try:
                        line = pipe_reader.readline()
                        if not line:
                            break
                        
                        # Filter out annoying FFmpeg/Qt warnings
                        ignore_patterns = [
                            "Late SEI is not implemented",
                            "If you want to help, upload a sample",
                            "No HW decoder found",
                            "Input #0, mov,mp4",
                            "Metadata:",
                            "major_brand",
                            "minor_version",
                            "compatible_brands",
                            "encoder",
                            "Duration:",
                            "Stream #0:",
                            "handler_name",
                            "vendor_id",
                            "libavutil",
                            "libavcodec",
                            "libavformat",
                            "libavdevice",
                            "libavfilter",
                            "libswscale",
                            "libswresample",
                        ]
                        
                        if any(pattern in line for pattern in ignore_patterns):
                            continue
                            
                        real_stderr.write(line)
                        real_stderr.flush()
                    except ValueError:
                        break
                    except Exception:
                        pass

def main():
    # Start stderr filter
    stderr_filter = StderrFilter()

    # Force software video decoding to prevent CUDA/hardware acceleration errors
    os.environ['QT_MEDIA_BACKEND'] = 'ffmpeg'
    os.environ['LIBVA_DRIVER_NAME'] = 'i965'  # Force software VA-API if available
    os.environ['VDPAU_DRIVER'] = 'va_gl'  # Fallback VDPAU driver
    
    # Suppress Qt Multimedia warnings
    os.environ["QT_LOGGING_RULES"] = "qt.multimedia*=false"
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Load custom fonts
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    if os.path.exists(font_dir):
        for font_file in os.listdir(font_dir):
            if font_file.endswith(".ttf"):
                QFontDatabase.addApplicationFont(os.path.join(font_dir, font_file))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

