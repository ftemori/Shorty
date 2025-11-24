from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

class InputWidget(QWidget):
    video_source_selected = pyqtSignal(str)  # Emits file path

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Drag & Drop Area
        self.drop_area = QLabel("Drag & Drop Video File Here\nor Click to Select")
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px solid #444;
                border-radius: 10px;
                color: #aaa;
                font-size: 16px;
                background-color: #252525;
            }
            QLabel:hover {
                border-color: #4a90e2;
                color: #fff;
                background-color: #2a2a2a;
            }
        """)
        self.drop_area.mousePressEvent = self.open_file_dialog
        layout.addWidget(self.drop_area)

    def open_file_dialog(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.mkv *.mov *.avi)"
        )
        if file_path:
            self.video_source_selected.emit(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            if f.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
                self.video_source_selected.emit(f)
                break # Just take the first one for now
