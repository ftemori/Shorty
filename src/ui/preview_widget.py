from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QFrame,
    QProgressBar,
    QSizePolicy,
    QScrollArea,
    QStackedWidget,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from src.ui.player_widget import VideoPlayer
import os


class ClipCard(QWidget):
    """A card widget representing a single analyzed clip with Generate/Download buttons."""
    generate_clicked = pyqtSignal(dict)  # Emits clip data
    download_clicked = pyqtSignal(dict)  # Emits clip data
    
    # Width of the info box
    BOX_WIDTH = 398
    BOX_HEIGHT_INITIAL = 100
    
    def __init__(self, clip_data, parent=None):
        super().__init__(parent)
        self.clip_data = clip_data
        self.generated = False
        self.output_path = None
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)
        
        # Left: Container with optional info text above and video box below
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # Info text container (hidden initially, shown above box after generation)
        self.info_text_container = QWidget()
        self.info_text_container.setFixedWidth(self.BOX_WIDTH)  # Match box width for center alignment
        info_text_layout = QVBoxLayout(self.info_text_container)
        info_text_layout.setContentsMargins(0, 0, 0, 5)
        info_text_layout.setSpacing(2)
        
        score = self.clip_data.get("score", 0)
        start = self.format_timestamp(self.clip_data.get("start", 0))
        end = self.format_timestamp(self.clip_data.get("end", 0))
        duration = self.clip_data.get("duration", self.clip_data.get("end", 0) - self.clip_data.get("start", 0))
        mins = int(duration // 60)
        secs = int(duration % 60)
        
        self.info_score_label = QLabel(f"{score:.2f}")
        self.info_score_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold; background: transparent;")
        self.info_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_text_layout.addWidget(self.info_score_label)
        
        self.info_time_label = QLabel(f"{start} - {end}")
        self.info_time_label.setStyleSheet("color: white; font-size: 13px; background: transparent;")
        self.info_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_text_layout.addWidget(self.info_time_label)
        
        # Duration label not shown above box - video player shows it
        
        self.info_text_container.hide()  # Hidden initially, text is inside blue box
        left_layout.addWidget(self.info_text_container)
        
        # Box container for the video/info box
        self.box_container = QWidget()
        self.box_container.setFixedSize(self.BOX_WIDTH, self.BOX_HEIGHT_INITIAL)
        box_container_layout = QVBoxLayout(self.box_container)
        box_container_layout.setContentsMargins(0, 0, 0, 0)
        box_container_layout.setSpacing(0)
        
        # Stacked widget to switch between info and video player
        self.stack = QStackedWidget()
        
        # Page 0: Info Box (blue rounded rectangle with text inside)
        self.info_box = QFrame()
        self.info_box.setStyleSheet("""
            QFrame {
                background-color: #6366f1;
                border-radius: 15px;
                border: none;
            }
        """)
        info_box_layout = QVBoxLayout(self.info_box)
        info_box_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_box_layout.setSpacing(2)
        
        # Labels inside blue box
        self.box_score_label = QLabel(f"{score:.2f}")
        self.box_score_label.setStyleSheet("color: white; font-size: 22px; font-weight: bold; background: transparent;")
        self.box_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_box_layout.addWidget(self.box_score_label)
        
        self.box_time_label = QLabel(f"{start} - {end}")
        self.box_time_label.setStyleSheet("color: white; font-size: 14px; background: transparent;")
        self.box_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_box_layout.addWidget(self.box_time_label)
        
        self.box_duration_label = QLabel(f"{mins}:{secs:02d}s")
        self.box_duration_label.setStyleSheet("color: white; font-size: 12px; background: transparent;")
        self.box_duration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_box_layout.addWidget(self.box_duration_label)
        
        self.stack.addWidget(self.info_box)
        
        # Page 1: Video Player (shown after generation)
        self.clip_player = VideoPlayer()
        self.stack.addWidget(self.clip_player)
        
        box_container_layout.addWidget(self.stack)
        
        # Progress Overlay (child of box_container, not in layout)
        self.progress_overlay = QWidget(self.box_container)
        self.progress_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0.92); border-radius: 15px;")
        self.progress_overlay.hide()
        
        overlay_layout = QVBoxLayout(self.progress_overlay)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label = QLabel("Generating...")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold; background: transparent; border: none;")
        overlay_layout.addWidget(self.progress_label)
        
        # Position overlay to cover the box
        self.progress_overlay.setGeometry(0, 0, self.BOX_WIDTH, self.BOX_HEIGHT_INITIAL)
        
        # Add box_container to left_layout
        left_layout.addWidget(self.box_container)
        left_layout.addStretch()
        
        layout.addWidget(left_container, alignment=Qt.AlignmentFlag.AlignTop)
        
        # Right: Buttons container - we need to add spacing at top to align with box (not text)
        self.btn_container = QWidget()
        btn_container_layout = QVBoxLayout(self.btn_container)
        btn_container_layout.setContentsMargins(0, 0, 0, 0)
        btn_container_layout.setSpacing(0)
        
        # Spacer that matches the info_text_container height (hidden initially)
        self.btn_top_spacer = QWidget()
        self.btn_top_spacer.setFixedHeight(0)  # No spacer initially since text is inside box
        btn_container_layout.addWidget(self.btn_top_spacer)
        
        # Buttons
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        
        # Generate Button
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.generate_btn.setFixedSize(140, 45)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7c7ff2;
            }
        """)
        self.generate_btn.clicked.connect(self.on_generate)
        btn_layout.addWidget(self.generate_btn)
        
        # Download Button (faded initially)
        self.download_btn = QPushButton("Download")
        self.download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.download_btn.setFixedSize(140, 45)
        self.set_download_faded(True)
        self.download_btn.clicked.connect(self.on_download)
        btn_layout.addWidget(self.download_btn)
        btn_layout.addStretch()
        
        btn_container_layout.addLayout(btn_layout)
        
        layout.addWidget(self.btn_container, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addStretch()
    
    def set_download_faded(self, faded):
        if faded:
            self.download_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(110, 231, 183, 0.4);
                    color: rgba(16, 185, 129, 0.6);
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)
            self.download_btn.setEnabled(False)
        else:
            self.download_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6ee7b7;
                    color: #10b981;
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #a7f3d0;
                }
            """)
            self.download_btn.setEnabled(True)
            self.download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def set_generate_faded(self, faded):
        if faded:
            self.generate_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(99, 102, 241, 0.4);
                    color: rgba(255, 255, 255, 0.5);
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)
            self.generate_btn.setEnabled(False)
        else:
            self.generate_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6366f1;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #7c7ff2;
                }
            """)
            self.generate_btn.setEnabled(True)
    
    def on_generate(self):
        # Show progress overlay
        self.progress_overlay.show()
        self.progress_overlay.raise_()
        self.progress_label.setText("Generating...")
        self.set_generate_faded(True)
        
        # Start simulated progress timer (since actual generation doesn't report progress)
        self._progress_value = 0
        from PyQt6.QtCore import QTimer
        self._progress_timer = QTimer()
        self._progress_timer.timeout.connect(self._simulate_progress)
        self._progress_timer.start(150)  # Update every 150ms
        
        self.generate_clicked.emit(self.clip_data)
    
    def _simulate_progress(self):
        """Simulate progress animation while generating (optimized for ~120s generation)."""
        if self._progress_value < 98:
            # Very slow progression to match ~120 second generation time
            # Timer runs every 150ms, so ~800 ticks for 120 seconds
            if self._progress_value < 20:
                self._progress_value += 0.15  # ~20 seconds to reach 20%
            elif self._progress_value < 40:
                self._progress_value += 0.12  # ~25 seconds to reach 40%
            elif self._progress_value < 60:
                self._progress_value += 0.10  # ~30 seconds to reach 60%
            elif self._progress_value < 75:
                self._progress_value += 0.08  # ~28 seconds to reach 75%
            elif self._progress_value < 85:
                self._progress_value += 0.05  # ~30 seconds to reach 85%
            elif self._progress_value < 92:
                self._progress_value += 0.03  # ~35 seconds to reach 92%
            else:
                self._progress_value += 0.01  # Very slow crawl after 92%
            self.progress_label.setText(f"Generating... {int(self._progress_value)}%")
    
    def update_progress(self, percent):
        """Update the generation progress overlay."""
        self._progress_value = percent
        self.progress_label.setText(f"Generating... {percent}%")
    
    def on_download(self):
        if self.output_path:
            self.download_clicked.emit({"clip": self.clip_data, "path": self.output_path})
    
    def mark_generated(self, output_path):
        """Called after generation completes to enable download and show video."""
        self.generated = True
        self.output_path = output_path
        
        # Stop progress timer
        if hasattr(self, '_progress_timer') and self._progress_timer:
            self._progress_timer.stop()
            self._progress_timer = None
        
        # Show 100% briefly then hide
        self.progress_label.setText("Generating... 100%")
        
        # Hide progress overlay
        self.progress_overlay.hide()
        
        # Show info text above the box (was hidden, inside blue box before)
        self.info_text_container.show()
        
        # Calculate the height of the info text container for button spacer
        # Approximately: 2 labels + spacing ≈ 45px (duration removed since player shows it)
        info_text_height = 45
        self.btn_top_spacer.setFixedHeight(info_text_height)
        
        # Calculate new height for 9:16 aspect ratio
        new_height = int(self.BOX_WIDTH * 16 / 9)
        
        # Resize the container to 9:16 aspect ratio
        self.box_container.setFixedSize(self.BOX_WIDTH, new_height)
        
        # Update overlay size (in case we need it again)
        self.progress_overlay.setGeometry(0, 0, self.BOX_WIDTH, new_height)
        
        # Style the info_box as black (it will be behind the player)
        self.info_box.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border-radius: 15px;
                border: none;
            }
        """)
        
        # Switch to video player and load the generated video
        self.stack.setCurrentIndex(1)
        self.clip_player.set_source(output_path)
        
        self.set_generate_faded(True)
        self.set_download_faded(False)
    
    @staticmethod
    def format_timestamp(seconds):
        seconds = max(0, int(seconds or 0))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

class VideoPreviewWidget(QWidget):
    analyze_clicked = pyqtSignal(str) # Emits file path
    generate_requested = pyqtSignal(str, list) # Emits file path + selected clips
    back_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_path = None
        self.analyzed_clips = []
        self.init_ui()

    def init_ui(self):
        # Simple layout - scrolling handled by main_window
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        # Container for video display area
        self.video_container = QWidget()
        video_container_layout = QVBoxLayout(self.video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.setSpacing(0)
        
        # Video Info Frame - shown by default when no video
        self.video_info_frame = QFrame()
        self.video_info_frame.setMinimumSize(640, 360) # Minimum 16:9
        self.video_info_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_info_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 12px;
                border: 1px solid #3a3a3a;
            }
        """)
        
        info_frame_layout = QVBoxLayout(self.video_info_frame)
        info_frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.info_label = QLabel("No video loaded")
        self.info_label.setStyleSheet("""
            QLabel {
                color: #ddd;
                font-size: 16px;
                background: transparent;
            }
        """)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setWordWrap(True)
        info_frame_layout.addWidget(self.info_label)
        
        video_container_layout.addWidget(self.video_info_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Video Player - hidden by default, shown when video loads
        self.player = VideoPlayer()
        self.player.setMinimumSize(640, 360)
        self.player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.player.hide()
        video_container_layout.addWidget(self.player, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.video_container, alignment=Qt.AlignmentFlag.AlignCenter)

        # Analyze Button
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.analyze_btn = QPushButton("🔎 Analize Shorts")
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                padding: 15px 30px;
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:disabled {
                background-color: #2c3e50;
                color: #888;
            }
        """)
        self.analyze_btn.clicked.connect(self.on_analyze)
        btn_layout.addWidget(self.analyze_btn)

        layout.addLayout(btn_layout)

        # Permanent Separator Line (70px under button)
        self.perm_separator = QFrame()
        self.perm_separator.setFrameShape(QFrame.Shape.HLine)
        self.perm_separator.setFrameShadow(QFrame.Shadow.Plain)
        self.perm_separator.setStyleSheet("background-color: #444; margin-top: 70px;")
        self.perm_separator.setFixedHeight(1)
        layout.addWidget(self.perm_separator)

        # Analysis Results Section (below separator)
        self.clips_section = QWidget()
        self.clips_section.setStyleSheet("background-color: rgb(20, 20, 20);")
        clips_section_layout = QVBoxLayout(self.clips_section)
        clips_section_layout.setContentsMargins(0, 20, 0, 0)
        clips_section_layout.setSpacing(0)
        
        # Progress indicators (shown during analysis)
        self.analysis_progress_container = QWidget()
        progress_layout = QVBoxLayout(self.analysis_progress_container)
        progress_layout.setContentsMargins(20, 10, 20, 10)
        
        self.analysis_status = QLabel("Run an analysis to preview clip ratings.")
        self.analysis_status.setStyleSheet("color: #aaa; font-size: 14px;")
        progress_layout.addWidget(self.analysis_status)
        
        self.analysis_progress_label = QLabel("")
        self.analysis_progress_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self.analysis_progress_label.setVisible(False)
        progress_layout.addWidget(self.analysis_progress_label)
        
        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setRange(0, 100)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_bar.setTextVisible(False)
        self.analysis_progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #333;
                border-radius: 4px;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #6366f1;
                border-radius: 4px;
            }
        """)
        self.analysis_progress_bar.setVisible(False)
        progress_layout.addWidget(self.analysis_progress_bar)
        
        clips_section_layout.addWidget(self.analysis_progress_container)
        
        # Container for clip cards (no inner scroll - whole page scrolls)
        self.clips_container = QWidget()
        self.clips_container.setStyleSheet("background-color: rgb(20, 20, 20);")
        self.clips_layout = QVBoxLayout(self.clips_container)
        self.clips_layout.setContentsMargins(10, 10, 10, 10)
        self.clips_layout.setSpacing(15)
        
        clips_section_layout.addWidget(self.clips_container)
        
        self.clips_section.setVisible(False)
        layout.addWidget(self.clips_section)
        
        # Store clip card references
        self.clip_cards = []

    def set_video(self, path):
        self.video_path = path
        filename = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        self.info_label.setText(f"File: {filename}\nSize: {size_mb:.1f} MB")
        
        # Hide info frame and show video player
        self.video_info_frame.hide()
        self.player.show()
        self.player.set_source(path)
        
        # Don't reset analysis here - clips persist until user explicitly analyzes new video

    def reset_analysis(self):
        self.analyzed_clips = []
        # Clear existing clip cards
        for card in self.clip_cards:
            card.deleteLater()
        self.clip_cards = []
        
        self.analysis_progress_label.setVisible(False)
        self.analysis_progress_bar.setVisible(False)
        self.analysis_progress_bar.setValue(0)
        self.analysis_status.setText("")
        self.clips_section.setVisible(False)
        self.set_analyze_busy(False)

    def begin_analysis(self):
        # Keep player visible during analysis
        # self.player.setVisible(False)
        
        # Clear existing clip cards
        for card in self.clip_cards:
            card.deleteLater()
        self.clip_cards = []
        
        self.clips_section.setVisible(True)
        self.analysis_status.setText("Analyzing video... This may take a moment.")
        self.analysis_progress_label.setText("Preparing...")
        self.analysis_progress_label.setVisible(True)
        self.analysis_progress_bar.setVisible(True)
        self.analysis_progress_bar.setRange(0, 0)  # Indeterminate until we know total

    def update_analysis_progress(self, current, total):
        if total:
            percent = int((current / total) * 100)
            self.analysis_progress_bar.setRange(0, 100)
            self.analysis_progress_bar.setValue(percent)
            self.analysis_progress_label.setText(f"Analyzing scenes {current}/{total} ({percent}%)")
        else:
            self.analysis_progress_bar.setRange(0, 0)
            self.analysis_progress_label.setText("Analyzing video...")

    def show_analysis(self, clips):
        # Keep player visible during analysis results
        # self.player.stop()  # Don't stop the player
        # self.player.setVisible(False)  # Keep player visible
        self.analysis_progress_label.setVisible(False)
        self.analysis_progress_bar.setVisible(False)
        self.analysis_status.setVisible(False)

        # Store the video path that was analyzed with each clip
        # This ensures clips generate from the correct video even if user imports a new one
        analyzed_video_path = self.video_path
        
        self.analyzed_clips = clips or []
        
        # Add the source video path to each clip's data
        for clip in self.analyzed_clips:
            clip["source_video_path"] = analyzed_video_path
        
        # Clear existing clip cards
        for card in self.clip_cards:
            card.deleteLater()
        self.clip_cards = []

        if not self.analyzed_clips:
            self.analysis_status.setText("No eligible clips were found between 15s and 60s.")
            self.analysis_status.setVisible(True)
        else:
            # Create clip cards for each analyzed clip
            for clip in self.analyzed_clips:
                card = ClipCard(clip)
                card.generate_clicked.connect(self.on_clip_generate)
                card.download_clicked.connect(self.on_clip_download)
                self.clips_layout.addWidget(card)
                self.clip_cards.append(card)

        self.clips_section.setVisible(True)
        self.set_analyze_busy(False)

    def on_clip_generate(self, clip_data):
        """Handle generate button click on a single clip card."""
        # Use the source video path stored in clip_data (from when analysis was done)
        # This ensures we generate from the correct video even if user imported a new one
        source_video = clip_data.get("source_video_path")
        if not source_video:
            # Fallback to current video path (shouldn't happen with new clips)
            source_video = self.video_path
        if not source_video:
            return
        # Stop player before generation
        if hasattr(self, 'player'):
            self.player.stop()
        self.generate_requested.emit(source_video, [clip_data])
    
    def on_clip_download(self, data):
        """Handle download button click - open file location."""
        import subprocess
        path = data.get("path")
        if path and os.path.exists(path):
            # Open file manager to the file location
            folder = os.path.dirname(path)
            subprocess.Popen(['xdg-open', folder])
    
    def mark_clip_generated(self, clip_data, output_path):
        """Mark a specific clip as generated (called from main window after generation)."""
        for card in self.clip_cards:
            if card.clip_data.get("index") == clip_data.get("index"):
                card.mark_generated(output_path)
                break
    
    def update_clip_progress(self, clip_data, percent):
        """Update the generation progress for a specific clip card."""
        for card in self.clip_cards:
            if card.clip_data.get("index") == clip_data.get("index"):
                card.update_progress(percent)
                break

    def on_analyze(self):
        # Don't stop player - keep it playable during and after analysis
        # if hasattr(self, 'player'):
        #     self.player.stop()
        if self.video_path:
            # Check if there are any generated clips that would be lost
            has_generated_clips = any(card.generated for card in self.clip_cards)
            
            if has_generated_clips:
                # Show confirmation dialog
                reply = QMessageBox.question(
                    self,
                    "Confirm Analysis",
                    "The generated shorts will be lost. Do you still want to continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply != QMessageBox.StandardButton.Yes:
                    return  # User cancelled, don't proceed
            
            # Reset analysis and proceed
            self.reset_analysis()
            self.begin_analysis()
            self.set_analyze_busy(True)
            self.analyze_clicked.emit(self.video_path)

    def on_back(self):
        self.back_clicked.emit()

    def set_analyze_busy(self, busy):
        if busy:
            self.analyze_btn.setText("Analyzing...")
            self.analyze_btn.setEnabled(False)
        else:
            self.analyze_btn.setText("🔎 Analize Shorts")
            self.analyze_btn.setEnabled(True)

    @staticmethod
    def format_timestamp(seconds):
        seconds = max(0, int(seconds or 0))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
