import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSlider, QHBoxLayout, QLabel, QGraphicsView, QGraphicsScene
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import Qt, QUrl, QTimer, pyqtSignal, QRectF
from PyQt6.QtGui import QPainter, QColor, QPalette, QPainterPath, QBrush, QPen

class ModernProgressBar(QSlider):
    """
    Ultra-thin progress bar like YouTube Shorts with circle handle on hover only
    """
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setFixedHeight(14)  # Height to accommodate the handle on hover
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style(hovered=False)
        
    def _update_style(self, hovered=False):
        if hovered:
            # Show red circle handle on hover
            handle_style = """
                QSlider::handle:horizontal {
                    background: #ff0000;
                    width: 10px;
                    height: 10px;
                    margin: -3px 0;
                    border-radius: 5px;
                    border: none;
                }
            """
        else:
            # Hide handle when not hovering (transparent, no size)
            handle_style = """
                QSlider::handle:horizontal {
                    background: transparent;
                    width: 0px;
                    height: 0px;
                    margin: 0px 0;
                    border: none;
                }
            """
        
        self.setStyleSheet(f"""
            QSlider {{
                background: transparent;
                border: none;
            }}
            QSlider::groove:horizontal {{
                background: rgba(255, 255, 255, 0.3);
                height: 4px;
                border: none;
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: #ff0000;
                height: 4px;
                border: none;
                border-radius: 2px;
            }}
            QSlider::add-page:horizontal {{
                background: rgba(255, 255, 255, 0.3);
                height: 4px;
                border: none;
                border-radius: 2px;
            }}
            {handle_style}
        """)
    
    def enterEvent(self, event):
        self._update_style(hovered=True)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self._update_style(hovered=False)
        super().leaveEvent(event)


class RoundedGraphicsView(QGraphicsView):
    """
    Custom QGraphicsView with rounded corners.
    This is the KEY to making rounded corners work with video!
    """
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.radius = 12
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("border: none; background: #1e1e1e;")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
    
    def drawBackground(self, painter, rect):
        """Draw black rounded background"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(rect, QBrush(QColor(0, 0, 0)))
    
    def drawForeground(self, painter, rect):
        """Draw the rounded corner mask OVER the video"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get the view's rectangle
        view_rect = self.viewport().rect()
        
        # Create full rectangle path
        full_path = QPainterPath()
        full_path.addRect(QRectF(view_rect))
        
        # Create rounded rectangle path
        rounded_path = QPainterPath()
        rounded_path.addRoundedRect(QRectF(view_rect), self.radius, self.radius)
        
        # Subtract to get corner areas
        corners = full_path.subtracted(rounded_path)
        
        # Transform from view coordinates to scene coordinates
        painter.save()
        painter.resetTransform()
        
        # Fill corners with window background color
        painter.fillPath(corners, QBrush(QColor(20, 20, 20)))
        
        # Draw border
        # Use the same rect as the mask to ensure curves match perfectly
        painter.setPen(QPen(QColor(20, 20, 20), 2))
        painter.drawRoundedRect(QRectF(view_rect), self.radius, self.radius)
        
        painter.restore()


class VideoPlayer(QWidget):
    """
    Modern video player with PROPER rounded corners using QGraphicsView.
    This is the correct approach for clipping video content.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setStyleSheet("""
            VideoPlayer {
                background-color: #1e1e1e;
                border-radius: 12px;
            }
        """)
        
        self.init_ui()
        
        # Media player setup
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        # Set video output to QGraphicsVideoItem instead of QVideoWidget
        self.media_player.setVideoOutput(self.video_item)
        
        # Connect signals
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.playbackStateChanged.connect(self.on_state_changed)
        self.media_player.errorOccurred.connect(self.on_error)
        
        # Default volume
        self.audio_output.setVolume(0.7)
        
        # Hover timer for controls
        self.controls_visible = True
        self.hide_timer = QTimer(self)
        self.hide_timer.timeout.connect(self.hide_controls)
        self.hide_timer.setSingleShot(True)

    def init_ui(self):
        """Initialize UI with QGraphicsView for proper video clipping"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create graphics scene
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        
        # Create QGraphicsVideoItem (this replaces QVideoWidget)
        self.video_item = QGraphicsVideoItem()
        self.video_item.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.scene.addItem(self.video_item)
        
        # Create custom graphics view with rounded corners
        self.view = RoundedGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.clicked.connect(self.toggle_playback)
        
        layout.addWidget(self.view)
        
        # Control bar overlay
        self.control_bar = QWidget(self)
        self.control_bar.setStyleSheet("background: transparent;")
        control_layout = QVBoxLayout(self.control_bar)
        control_layout.setContentsMargins(12, 20, 12, 12)
        control_layout.setSpacing(8)
        
        # Time label
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 11px;
                background: transparent;
                font-weight: bold;
            }
        """)
        control_layout.addWidget(self.time_label)
        
        # Progress bar
        self.progress_bar = ModernProgressBar(self)
        self.progress_bar.sliderMoved.connect(self.set_position)
        self.progress_bar.sliderPressed.connect(self.on_slider_pressed)
        self.progress_bar.sliderReleased.connect(self.on_slider_released)
        control_layout.addWidget(self.progress_bar)

    def resizeEvent(self, event):
        """Update video and control positioning on resize"""
        super().resizeEvent(event)
        
        w = self.width()
        h = self.height()
        
        # Set scene rect to match widget size
        self.scene.setSceneRect(0, 0, w, h)
        
        # Set video item to fill the scene
        self.video_item.setSize(QRectF(0, 0, w, h).size())
        
        # Center the video item in the scene
        self.video_item.setPos(0, 0)
        
        # Don't use fitInView - let the scene size control it
        self.view.setSceneRect(0, 0, w, h)
        
        # Position control bar
        bar_height = 80
        self.control_bar.setGeometry(0, h - bar_height, w, bar_height)
        self.control_bar.raise_()
        
        # Force view to redraw foreground (rounded corners)
        self.view.viewport().update()

    def set_source(self, file_path):
        """Set video source and prepare for playback"""
        self.stop()
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        abs_path = os.path.abspath(file_path)
        url = QUrl.fromLocalFile(abs_path)
        self.media_player.setSource(url)
        
        # Auto-play when video is loaded
        self.media_player.play()

    def toggle_playback(self):
        """Toggle between play and pause"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def position_changed(self, position):
        """Update progress bar when video position changes"""
        if not self.progress_bar.isSliderDown():
            self.progress_bar.setValue(position)
        self.update_time_label(position, self.media_player.duration())

    def duration_changed(self, duration):
        """Set progress bar range when video duration is known"""
        self.progress_bar.setRange(0, duration)
        self.update_time_label(self.media_player.position(), duration)

    def set_position(self, position):
        """Seek to specific position"""
        self.media_player.setPosition(position)

    def on_slider_pressed(self):
        """Pause updates when user is dragging slider"""
        pass

    def on_slider_released(self):
        """Resume updates when user releases slider"""
        self.media_player.setPosition(self.progress_bar.value())

    def update_time_label(self, position, duration):
        """Update time display"""
        current = self.format_time(position)
        total = self.format_time(duration)
        self.time_label.setText(f"{current} / {total}")

    @staticmethod
    def format_time(ms):
        """Format milliseconds to MM:SS or HH:MM:SS"""
        seconds = ms // 1000
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def stop(self):
        """Stop playback and clear source"""
        self.media_player.stop()
        self.media_player.setSource(QUrl())

    def on_state_changed(self, state):
        """Handle playback state changes"""
        pass

    def on_error(self, error, error_string):
        """Handle media player errors"""
        print(f"Media player error: {error_string}")

    def show_controls(self):
        """Show control bar"""
        self.control_bar.show()
        self.controls_visible = True
        self.hide_timer.start(3000)

    def hide_controls(self):
        """Hide control bar (except when paused)"""
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            return
        self.controls_visible = False

    def enterEvent(self, event):
        """Show controls on hover"""
        self.show_controls()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Start hide timer when mouse leaves"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.hide_timer.start(1000)
        super().leaveEvent(event)
