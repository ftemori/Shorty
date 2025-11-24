from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QPushButton, QFrame, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage
import requests
import re

class ImageLoader(QThread):
    loaded = pyqtSignal(str, QPixmap)

    def __init__(self, url, video_id):
        super().__init__()
        self.url = url
        self.video_id = video_id

    def run(self):
        try:
            response = requests.get(self.url, timeout=5)
            if response.status_code == 200:
                image = QImage()
                image.loadFromData(response.content)
                pixmap = QPixmap(image)
                self.loaded.emit(self.video_id, pixmap)
        except Exception:
            pass

class VideoSelectionWidget(QWidget):
    video_selected = pyqtSignal(str) # Emits video URL
    channel_url_entered = pyqtSignal(str) # Emits channel URL to fetch videos

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.loaders = []

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.container = QFrame()
        self.container.setStyleSheet("""
            QFrame {
                border: 2px solid #444;
                border-radius: 10px;
                background-color: #252525;
            }
        """)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        
        # URL Input Section
        input_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste YouTube Channel URL...")
        self.url_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border-radius: 5px;
                background-color: #333;
                color: white;
                border: 1px solid #444;
            }
            QLineEdit:focus {
                border: 1px solid #4a90e2;
            }
        """)
        input_layout.addWidget(self.url_input)

        self.load_btn = QPushButton("Load")
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
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
        input_layout.addWidget(self.load_btn)
        layout.addLayout(input_layout)

        # Scroll Area for List
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        self.list_container = QWidget()
        self.list_container.setStyleSheet("background-color: transparent; border: none;") # Ensure transparent
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setSpacing(10)
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.list_container)
        
        layout.addWidget(scroll)

    def on_load_clicked(self):
        url = self.url_input.text().strip()
        if not url:
            return
            
        # Basic Validation
        youtube_regex = (
            r'(https?://)?(www\.)?'
            r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        channel_regex = r'(https?://)?(www\.)?youtube\.com/(channel/|c/|user/|@)[^/]+'
        
        if not (re.match(youtube_regex, url) or re.match(channel_regex, url)):
             QMessageBox.warning(self, "Invalid URL", "Please enter a valid YouTube URL.")
             return

        self.channel_url_entered.emit(url)

    def populate(self, videos):
        # Clear existing
        for i in reversed(range(self.list_layout.count())): 
            item = self.list_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        self.loaders.clear()
        
        for video in videos:
            if not video: continue
            
            title = video.get('title', 'Unknown Title')
            url = video.get('url') or video.get('webpage_url') or video.get('original_url')
            video_id = video.get('id')
            if not url and video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
            
            thumbnails = video.get('thumbnails', [])
            thumb_url = thumbnails[-1]['url'] if thumbnails else None
            
            # Extract date if available (upload_date is usually YYYYMMDD)
            upload_date = video.get('upload_date', '')
            if upload_date and len(upload_date) == 8:
                formatted_date = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
            else:
                formatted_date = "Unknown Date"

            card = self.create_video_list_item(title, thumb_url, url, video_id, formatted_date)
            self.list_layout.addWidget(card)

    def create_video_list_item(self, title_text, thumb_url, video_url, video_id, date_text):
        card = QFrame()
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setFixedHeight(80) # Fixed height for list item
        card.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 5px;
                padding: 5px;
            }
            QFrame:hover {
                background-color: #444;
                border: 1px solid #4a90e2;
            }
        """)
        card.mousePressEvent = lambda e: self.video_selected.emit(video_url)
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Thumbnail
        thumb_label = QLabel()
        thumb_label.setFixedSize(120, 68) # 16:9 approx
        thumb_label.setStyleSheet("background-color: #000; border-radius: 3px;")
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(thumb_label)
        
        if thumb_url:
            loader = ImageLoader(thumb_url, video_id)
            loader.loaded.connect(lambda vid, pix: self.set_thumbnail(thumb_label, pix))
            loader.start()
            self.loaders.append(loader)
        else:
            thumb_label.setText("No Image")
        
        # Info Layout
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(5, 0, 0, 0)
        
        # Title
        name = QLabel(title_text)
        name.setWordWrap(True)
        name.setStyleSheet("color: white; font-size: 14px; font-weight: bold; background-color: transparent; border: none;")
        info_layout.addWidget(name)
        
        # Date
        date = QLabel(date_text)
        date.setStyleSheet("color: #aaa; font-size: 12px; background-color: transparent; border: none;")
        info_layout.addWidget(date)
        
        layout.addLayout(info_layout)
        
        return card

    def set_thumbnail(self, label, pixmap):
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation))
