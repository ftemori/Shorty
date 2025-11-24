import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSlider
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtGui import QPainter, QColor

class ClickableVideoWidget(QVideoWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_player = parent

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent_player.toggle_playback()
        super().mousePressEvent(event)

class ThinProgressBar(QSlider):
    """
    Ultra-thin progress bar like TikTok/YouTube Shorts
    """
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setFixedHeight(4)  # Very thin
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QSlider {
                background: transparent;
                border: none;
            }
            QSlider::groove:horizontal {
                background: rgba(255, 255, 255, 0.3);
                height: 4px;
                border: none;
            }
            QSlider::sub-page:horizontal {
                background: rgba(255, 255, 255, 0.95);
                height: 4px;
                border: none;
            }
            QSlider::add-page:horizontal {
                background: rgba(255, 255, 255, 0.3);
                height: 4px;
                border: none;
            }
            QSlider::handle:horizontal {
                background: white;
                width: 0px;
                height: 0px;
                margin: 0px;
                border: none;
            }
            QSlider::handle:horizontal:hover {
                background: white;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)

class VideoPlayer(QWidget):
    """
    Modern video player styled like TikTok/YouTube Shorts
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.playbackStateChanged.connect(self.on_state_changed)
        
        # Default volume
        self.audio_output.setVolume(0.7)
        
        # Auto-hide timer for progress bar
        self.hide_timer = QTimer(self)
        self.hide_timer.timeout.connect(self.maybe_hide_progress)
        self.hide_timer.start(3000)  # Check every 3 seconds

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video Area (fills entire space)
        self.video_widget = ClickableVideoWidget(self)
        self.video_widget.setStyleSheet("""
            QVideoWidget {
                background-color: #000000;
            }
        """)
        layout.addWidget(self.video_widget, stretch=1)

        # Thin progress bar overlay at the bottom
        # Create container for absolute positioning
        self.progress_container = QWidget(self)
        self.progress_container.setStyleSheet("background: transparent;")
        progress_layout = QVBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(0)
        
        self.progress_bar = ThinProgressBar(self)
        self.progress_bar.sliderMoved.connect(self.set_position)
        progress_layout.addWidget(self.progress_bar)
        
        # Position progress bar at the bottom
        self.progress_container.setGeometry(0, 0, self.width(), 4)
        self.progress_container.raise_()
        self.progress_container.show()
        
        # Make progress bar float at bottom
        self.progress_bar.setFixedHeight(4)

    def resizeEvent(self, event):
        """Keep progress bar at bottom when window resizes"""
        super().resizeEvent(event)
        if hasattr(self, 'progress_container'):
            # Position at bottom
            y_pos = self.height() - 4
            self.progress_container.setGeometry(0, y_pos, self.width(), 4)

    def set_source(self, file_path):
        self.stop()
        if not os.path.exists(file_path):
            return
        abs_path = os.path.abspath(file_path)
        self.media_player.setSource(QUrl.fromLocalFile(abs_path))
        self.media_player.play()

    def toggle_playback(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
        
        # Show progress bar briefly when toggling
        self.show_progress_briefly()

    def on_state_changed(self, state):
        """Show progress bar when paused, hide when playing"""
        if state == QMediaPlayer.PlaybackState.PausedState:
            self.progress_container.show()
        elif state == QMediaPlayer.PlaybackState.PlayingState:
            # Hide after a delay
            QTimer.singleShot(2000, self.maybe_hide_progress)

    def show_progress_briefly(self):
        """Show progress bar for 2 seconds"""
        self.progress_container.show()
        QTimer.singleShot(2000, self.maybe_hide_progress)

    def maybe_hide_progress(self):
        """Hide progress bar if playing"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            # Keep it visible, TikTok/Shorts style (always show thin line)
            pass  # Actually keep it visible always for TikTok style

    def position_changed(self, position):
        if not self.progress_bar.isSliderDown():
            self.progress_bar.setValue(position)

    def duration_changed(self, duration):
        self.progress_bar.setRange(0, duration)

    def set_position(self, position):
        self.media_player.setPosition(position)
        
    def stop(self):
        self.media_player.stop()
        self.media_player.setSource(QUrl())

    def enterEvent(self, event):
        """Show progress bar on hover"""
        self.progress_container.show()
        super().enterEvent(event)
