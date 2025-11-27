from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QPushButton, QFrame, QLineEdit, QMessageBox, QStackedLayout, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QPainter, QColor
import requests
import re
import html

class LoadingSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(80)

    def rotate(self):
        self.angle = (self.angle + 30) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        painter.translate(w / 2, h / 2)
        
        for i in range(12):
            painter.rotate(30)
            # Calculate opacity: The "head" is at self.angle.
            # We want the trail to fade out behind it.
            # self.angle corresponds to one of the 12 positions (0, 30, 60...)
            # Let's say angle=0. We want i=0 to be opaque? Or i=11?
            # Let's just use a relative index.
            
            current_step = self.angle / 30
            diff = (i - current_step) % 12
            opacity = 1.0 - (diff / 12.0)
            if opacity < 0: opacity = 0
            
            # Use white/grey color
            color = QColor(200, 200, 200)
            color.setAlphaF(opacity)
            
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            
            # Draw pill
            painter.drawRoundedRect(-3, -25, 6, 14, 3, 3)

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

class VideoCard(QFrame):
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(120)
        self.setStyleSheet("""
            VideoCard {
                background-color: #333;
                border-radius: 5px;
                padding: 0px;
            }
            VideoCard:hover {
                background-color: #444;
                border: 1px solid #4a90e2;
            }
        """)
        
        # Content Layout
        self.content_layout = QHBoxLayout(self)
        # Change the first value (left margin) from 5 to something larger, like 15 or 20
        self.content_layout.setContentsMargins(10, 5, 5, 5)
        
        # Overlay (child of self, not in layout)
        self.overlay = QWidget(self)
        self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0.85); border-radius: 5px;")
        self.overlay.hide()
        
        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label = QLabel("Downloading... 0%")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; background: transparent; border: none;")
        overlay_layout.addWidget(self.progress_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.resize(self.size())
        self.overlay.raise_()

class VideoSelectionWidget(QWidget):
    video_selected = pyqtSignal(str) # Emits video URL
    channel_url_entered = pyqtSignal(str) # Emits channel URL to fetch videos

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.loaders = []
        self.cards = {} # Map url -> card_widget

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.container = QFrame()
        self.container.setStyleSheet("""
            QFrame {
                border: none;
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

        # Stack for List vs Loading
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Page 1: Scroll Area (List)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: black;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #333;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #444;
            }
            QScrollBar::add-line:vertical {
                height: 0px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            QScrollBar::sub-line:vertical {
                height: 0px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        self.list_container = QWidget()
        self.list_container.setStyleSheet("background-color: transparent; border: none;") # Ensure transparent
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setSpacing(10)
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.list_container)
        self.stack.addWidget(scroll)
        
        # Page 2: Loading
        loading_page = QWidget()
        loading_page.setStyleSheet("background-color: transparent;")
        loading_layout = QVBoxLayout(loading_page)
        loading_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinner = LoadingSpinner()
        loading_layout.addWidget(self.spinner)
        self.stack.addWidget(loading_page)

    def start_loading(self):
        self.stack.setCurrentIndex(1)
        self.load_btn.setEnabled(False)
        self.url_input.setEnabled(False)

    def stop_loading(self):
        self.stack.setCurrentIndex(0)
        self.load_btn.setEnabled(True)
        self.url_input.setEnabled(True)

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

        self.start_loading()
        self.channel_url_entered.emit(url)

    def populate(self, videos):
        self.stop_loading()
        # Clear existing
        for i in reversed(range(self.list_layout.count())): 
            item = self.list_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        self.loaders.clear()
        self.cards.clear()
        
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
            upload_date = video.get('upload_date')
            if not upload_date and video.get('release_date'):
                upload_date = video.get('release_date')

            if upload_date and len(upload_date) == 8:
                formatted_date = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
            elif video.get('timestamp'):
                import datetime
                ts = video.get('timestamp')
                formatted_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            else:
                # Try to parse from other fields if available, or leave as Unknown
                formatted_date = "Unknown Date"
            
            channel_name = video.get('uploader') or video.get('channel') or "Unknown Channel"

            card = self.create_video_list_item(title, thumb_url, url, video_id, formatted_date, channel_name)
            self.list_layout.addWidget(card)

    def create_video_list_item(self, title_text, thumb_url, video_url, video_id, date_text, channel_text):
        card = VideoCard()
        card.clicked.connect(lambda: self.video_selected.emit(video_url))
        
        # Thumbnail
        thumb_label = QLabel()
        thumb_label.setFixedSize(180, 102) # 16:9 approx, 1.5x size
        thumb_label.setStyleSheet("background-color: #000; border-radius: 3px;")
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card.content_layout.addWidget(thumb_label)
        
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
        info_layout.setSpacing(2) # Reduced spacing between Title, Channel, Date
        
        # Title
        name = QLabel()
        name.setWordWrap(True)
        
        # Set font programmatically for better fallback support
        font = QFont()
        # Expanded font list including FreeSans and standard Linux fonts
        font.setFamilies([
            "Segoe UI", 
            "Helvetica Neue", 
            "Arial", 
            "Noto Sans Arabic", 
            "Noto Sans", 
            "FreeSans", 
            "DejaVu Sans", 
            "Liberation Sans", 
            "sans-serif"
        ])
        font.setBold(True)
        font.setPixelSize(14)
        name.setFont(font)
        
        # Use HTML to control line height (80% is tighter)
        escaped_title = html.escape(title_text)
        name.setText(f"<p style='line-height: 80%; margin: 0; color: white;'>{escaped_title}</p>")
        
        name.setStyleSheet("""
            background-color: transparent; 
            border: none;
        """)
        info_layout.addWidget(name)
        
        # Channel Name
        channel = QLabel(channel_text)
        channel.setStyleSheet("color: #aaa; font-size: 12px; font-weight: bold; background-color: transparent; border: none;")
        info_layout.addWidget(channel)
        
        # Date
        date = QLabel(date_text)
        date.setStyleSheet("color: #888; font-size: 12px; background-color: transparent; border: none;")
        info_layout.addWidget(date)
        
        card.content_layout.addLayout(info_layout)
        
        self.cards[video_url] = card
        
        return card

    def set_thumbnail(self, label, pixmap):
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation))

    def set_download_progress(self, url, message):
        if url in self.cards:
            card = self.cards[url]
            card.overlay.show()
            card.progress_label.setText(message)

    def hide_download_progress(self, url):
        if url in self.cards:
            card = self.cards[url]
            card.overlay.hide()
