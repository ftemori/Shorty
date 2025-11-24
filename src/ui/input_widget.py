from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, 
    QFrame, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

class InputWidget(QWidget):
    video_source_selected = pyqtSignal(str)  # Emits URL or file path

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title = QLabel("Add Content")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)

        # URL Input Section
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste YouTube Channel or Video URL...")
        self.url_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border-radius: 5px;
                background-color: #333;
                color: white;
                border: 1px solid #444;
            }
            QLineEdit:focus {
                border: 1px solid #4a90e2;
            }
        """)
        url_layout.addWidget(self.url_input)

        self.load_btn = QPushButton("Load")
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
        self.load_btn.clicked.connect(self.on_load_clicked)
        url_layout.addWidget(self.load_btn)
        layout.addLayout(url_layout)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setStyleSheet("background-color: #444;")
        layout.addWidget(divider)

        # Drag & Drop Area
        self.drop_area = QLabel("Drag & Drop Video File Here\nor Click to Select")
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #555;
                border-radius: 10px;
                padding: 40px;
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
        # Make label clickable effectively by wrapping or event filter, 
        # but for simplicity we'll add a transparent button or just mousePressEvent
        self.drop_area.mousePressEvent = self.open_file_dialog
        layout.addWidget(self.drop_area)

    def on_load_clicked(self):
        url = self.url_input.text().strip()
        if url:
            self.video_source_selected.emit(url)

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
