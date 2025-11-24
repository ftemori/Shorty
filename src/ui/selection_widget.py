from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGridLayout, QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage
import requests

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

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.loaders = []

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("Select a Video")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(15)
        scroll.setWidget(self.grid_container)
        
        layout.addWidget(scroll)
        
        # Back Button
        self.back_btn = QPushButton("Back")
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                background-color: #555;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        layout.addWidget(self.back_btn)

    def populate(self, videos):
        # Clear existing
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)
        self.loaders.clear()
            
        row = 0
        col = 0
        max_cols = 3
        
        for video in videos:
            if not video: continue
            
            title = video.get('title', 'Unknown Title')
            url = video.get('url') or video.get('webpage_url') or video.get('original_url')
            # yt-dlp extract_flat often gives 'url' as the video ID or partial url, need to construct full if needed
            # usually 'url' in flat extraction is the video URL or ID.
            # If it's just ID, we might need to prepend youtube.com
            
            video_id = video.get('id')
            if not url and video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
            
            thumbnails = video.get('thumbnails', [])
            thumb_url = thumbnails[-1]['url'] if thumbnails else None
            
            card = self.create_video_card(title, thumb_url, url, video_id)
            self.grid_layout.addWidget(card, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def create_video_card(self, title_text, thumb_url, video_url, video_id):
        card = QFrame()
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.setStyleSheet("""
            QFrame {
                background-color: #333;
                border-radius: 10px;
                padding: 10px;
            }
            QFrame:hover {
                background-color: #444;
                border: 1px solid #4a90e2;
            }
        """)
        card.mousePressEvent = lambda e: self.video_selected.emit(video_url)
        
        layout = QVBoxLayout(card)
        
        # Thumbnail
        thumb_label = QLabel()
        thumb_label.setFixedSize(200, 112) # 16:9 approx
        thumb_label.setStyleSheet("background-color: #000; border-radius: 5px;")
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(thumb_label)
        
        if thumb_url:
            loader = ImageLoader(thumb_url, video_id)
            loader.loaded.connect(lambda vid, pix: self.set_thumbnail(thumb_label, pix))
            loader.start()
            self.loaders.append(loader)
        else:
            thumb_label.setText("No Image")
        
        # Title
        name = QLabel(title_text)
        name.setWordWrap(True)
        name.setStyleSheet("color: white; font-size: 12px; margin-top: 5px; background-color: transparent; border: none;")
        layout.addWidget(name)
        
        return card

    def set_thumbnail(self, label, pixmap):
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation))
