from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPalette, QColor, QIcon
from src.ui.input_widget import InputWidget
from src.ui.selection_widget import VideoSelectionWidget
from src.ui.preview_widget import VideoPreviewWidget
from src.core.worker import Worker
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shorty")
        self.resize(1200, 900)
        self.apply_dark_theme()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Main scroll area for entire page
        self.main_scroll = QScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setStyleSheet("""
            QScrollArea {
                background-color: rgb(20, 20, 20);
                border: none;
            }
            QScrollBar:vertical {
                background-color: rgb(20, 20, 20);
                width: 8px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #444;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Scroll content widget
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: rgb(20, 20, 20);")
        
        # Content Layout
        self.content_layout = QVBoxLayout(scroll_content)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(20)
        
        self.main_scroll.setWidget(scroll_content)
        self.main_layout.addWidget(self.main_scroll)

        # 2. Top Section (Split 50/50)
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(20)
        
        # Left: Upload Box
        self.input_widget = InputWidget()
        self.input_widget.video_source_selected.connect(self.handle_local_file)
        # Style it to look like a box
        self.input_widget.setStyleSheet("background-color: #252525; border-radius: 10px;")
        top_layout.addWidget(self.input_widget)
        
        # Right: YT Box
        self.selection_widget = VideoSelectionWidget()
        self.selection_widget.channel_url_entered.connect(self.fetch_channel_videos)
        self.selection_widget.video_selected.connect(self.start_download)
        self.selection_widget.setStyleSheet("background-color: #252525; border-radius: 10px;")
        top_layout.addWidget(self.selection_widget)
        
        # Set equal width
        top_layout.setStretch(0, 1)
        top_layout.setStretch(1, 1)
        
        # Fixed height for top section (approx enough for list + scroll)
        top_section.setMinimumHeight(250)
        # top_section.setFixedHeight(300) 
        self.content_layout.addWidget(top_section, 1)

        # 3. Middle Section: Video Player (Centered, 60% width)
        # We use a container to center it
        player_container = QWidget()
        player_layout = QHBoxLayout(player_container)
        player_layout.setContentsMargins(0, 50, 0, 0) # 50px margin top
        # player_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.preview_widget = VideoPreviewWidget()
        
        # Connect signals
        self.preview_widget.analyze_clicked.connect(self.start_analysis)
        self.preview_widget.generate_requested.connect(self.start_generation)
        # back_clicked is probably not needed anymore as we are single page, 
        # but maybe it resets the player?
        self.preview_widget.back_clicked.connect(self.reset_player)

        player_layout.addWidget(self.preview_widget)
        self.content_layout.addWidget(player_container, 3)
        
        # 4. Bottom Section: Accounts Button
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        # self.accounts_btn = QPushButton("Accounts")
        # self.accounts_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # self.accounts_btn.setStyleSheet("""
        #     QPushButton {
        #         background-color: #333;
        #         color: white;
        #         border: 1px solid #555;
        #         padding: 8px 20px;
        #         border-radius: 4px;
        #     }
        #     QPushButton:hover {
        #         background-color: #444;
        #     }
        # """)
        # self.accounts_btn.clicked.connect(self.open_settings)
        # bottom_layout.addWidget(self.accounts_btn)

        self.content_layout.addLayout(bottom_layout)

        # Dictionary to track active generation workers by clip index
        # This allows multiple clips to generate simultaneously
        self.generation_workers = {}
        


    def handle_local_file(self, file_path):
        self.preview_widget.set_video(file_path)

    def fetch_channel_videos(self, url):
        # We can show a loading state in the selection widget if we want
        # For now, just run the worker
        self.worker = Worker("download_channel", url=url)
        self.worker.finished.connect(self.on_channel_fetched)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_channel_fetched(self, videos):
        if not videos:
            QMessageBox.warning(self, "No Videos", "No videos found for this channel/URL.")
            return
        self.selection_widget.populate(videos)

    def start_download(self, url):
        # Show loading indicator?
        # Maybe update the preview widget to show "Downloading..."
        self.preview_widget.info_label.setText("Downloading video...")
        self.current_download_url = url
        
        self.worker = Worker("download_video", url=url)
        self.worker.finished.connect(self.on_download_finished)
        self.worker.progress.connect(self.update_download_progress)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def update_download_progress(self, msg):
        if hasattr(self, 'current_download_url'):
            # Ignore "Downloading video: URL" messages
            if "Downloading video:" in msg:
                self.selection_widget.set_download_progress(self.current_download_url, "Starting download...")
                return
                
            # Parse "Downloading: 50%" to "Downloading... 50%"
            if "Downloading:" in msg:
                percent = msg.replace("Downloading:", "").replace("%", "").strip()
                formatted_msg = f"Downloading... {percent}%"
                self.selection_widget.set_download_progress(self.current_download_url, formatted_msg)
            else:
                # Only show if it looks like progress or status, not URL
                if "http" not in msg:
                    self.selection_widget.set_download_progress(self.current_download_url, msg)

    def on_download_finished(self, file_path):
        if hasattr(self, 'current_download_url'):
            self.selection_widget.hide_download_progress(self.current_download_url)
        self.preview_widget.set_video(file_path)

    def start_analysis(self, file_path):
        self.worker = Worker("analyze_file", video_path=file_path)
        self.worker.analysis_progress.connect(self.preview_widget.update_analysis_progress)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_analysis_finished(self, clips):
        self.preview_widget.show_analysis(clips)

    def start_generation(self, file_path, selected_clips):
        # Support concurrent generation - each clip gets its own worker
        for clip in selected_clips:
            clip_index = clip.get("index")
            
            # Skip if this clip is already being generated
            if clip_index in self.generation_workers:
                continue
            
            # Create a worker for this specific clip
            worker = Worker("generate_clips", video_path=file_path, clips=[clip])
            
            # Store reference to the clip in the worker for callbacks
            worker.generating_clip = clip
            
            # Connect signals with lambda to capture clip reference
            worker.progress.connect(lambda msg, c=clip: self.update_generation_progress(msg, c))
            worker.finished.connect(lambda paths, c=clip: self.on_generation_finished(paths, c))
            worker.error.connect(lambda err, c=clip: self.on_generation_error(err, c))
            
            # Track the worker
            self.generation_workers[clip_index] = worker
            worker.start()
    
    def update_generation_progress(self, msg, clip):
        """Handle generation progress updates for a specific clip."""
        if msg.startswith("GENERATION_PROGRESS:"):
            try:
                percent = int(msg.split(":")[1])
                # Update the clip card's progress overlay
                self.preview_widget.update_clip_progress(clip, percent)
            except (ValueError, IndexError):
                pass
        
    def on_generation_finished(self, paths, clip):
        """Handle generation completion for a specific clip."""
        # Remove worker from tracking
        clip_index = clip.get("index")
        if clip_index in self.generation_workers:
            del self.generation_workers[clip_index]
        
        # Mark the clip as generated with the output path
        if paths:
            output_path = paths[0] if isinstance(paths, list) else paths
            self.preview_widget.mark_clip_generated(clip, output_path)
    
    def on_generation_error(self, error_msg, clip):
        """Handle generation error for a specific clip."""
        # Remove worker from tracking
        clip_index = clip.get("index")
        if clip_index in self.generation_workers:
            del self.generation_workers[clip_index]
        
        QMessageBox.critical(self, "Generation Error", f"Error generating clip:\n{error_msg}")

    def reset_player(self):
        # Reset logic if needed
        pass

    def on_processing_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")
        self.preview_widget.set_analyze_busy(False)
        self.selection_widget.stop_loading()

    def open_settings(self):
        QMessageBox.information(self, "Accounts", "Account settings placeholder.\n\nHere you would connect TikTok, YouTube, and Instagram accounts.")

    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(20, 20, 20))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(20, 20, 20))
        self.setPalette(palette)
