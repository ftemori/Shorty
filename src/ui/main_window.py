from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor
from src.ui.input_widget import InputWidget
from src.ui.results_widget import ResultsWidget
from src.ui.selection_widget import VideoSelectionWidget
from src.ui.preview_widget import VideoPreviewWidget
from src.core.worker import Worker
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shorty - AI Shorts Generator")
        self.resize(1200, 800)
        self.apply_dark_theme()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            background-color: #202020;
            border-bottom: 1px solid #333;
        """)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel("Shorty")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white; border: none;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        settings_btn = QPushButton("Accounts")
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                padding: 5px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #444;
            }
        """)
        settings_btn.clicked.connect(self.open_settings)
        header_layout.addWidget(settings_btn)
        
        self.main_layout.addWidget(header_widget)
        
        # Content Area (Stacked)
        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)
        
        # Page 0: Input
        self.input_page = InputWidget()
        self.input_page.video_source_selected.connect(self.handle_source_input)
        self.stack.addWidget(self.input_page)
        
        # Page 1: Loading
        self.loading_page = QLabel("Initializing...")
        self.loading_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_page.setStyleSheet("font-size: 24px; color: #888;")
        self.stack.addWidget(self.loading_page)
        
        # Page 2: Selection
        self.selection_page = VideoSelectionWidget()
        self.selection_page.video_selected.connect(self.start_download)
        self.selection_page.back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.stack.addWidget(self.selection_page)
        
        # Page 3: Preview
        self.preview_page = VideoPreviewWidget()
        self.preview_page.analyze_clicked.connect(self.start_analysis)
        self.preview_page.generate_requested.connect(self.start_generation)
        self.preview_page.back_clicked.connect(lambda: self.stack.setCurrentIndex(2))
        self.stack.addWidget(self.preview_page)
        
        # Page 4: Results
        self.results_page = ResultsWidget()
        self.stack.addWidget(self.results_page)
        
    def handle_source_input(self, source):
        if os.path.exists(source):
            # Local file, go straight to preview
            self.show_preview(source)
        else:
            # Assume URL, fetch channel videos
            self.fetch_channel_videos(source)

    def fetch_channel_videos(self, url):
        self.stack.setCurrentIndex(1) # Show loading
        self.loading_page.setText("Fetching channel videos...")
        
        self.worker = Worker("download_channel", url=url)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_channel_fetched)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_channel_fetched(self, videos):
        if not videos:
            QMessageBox.warning(self, "No Videos", "No videos found for this channel/URL.")
            self.stack.setCurrentIndex(0)
            return
            
        self.selection_page.populate(videos)
        self.stack.setCurrentIndex(2) # Show selection

    def start_download(self, url):
        self.stack.setCurrentIndex(1) # Show loading
        self.loading_page.setText(f"Downloading video...")
        
        self.worker = Worker("download_video", url=url)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.show_preview)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def show_preview(self, file_path):
        self.preview_page.set_video(file_path)
        self.stack.setCurrentIndex(3) # Show preview

    def start_analysis(self, file_path):
        self.preview_page.begin_analysis()
        self.worker = Worker("analyze_file", video_path=file_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.analysis_progress.connect(self.preview_page.update_analysis_progress)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_analysis_finished(self, clips):
        self.preview_page.show_analysis(clips)
        self.stack.setCurrentIndex(3)

    def start_generation(self, file_path, selected_clips=None):
        if not selected_clips:
            QMessageBox.warning(self, "No Clips Selected", "Please choose at least one clip to generate.")
            self.stack.setCurrentIndex(3)
            return
        self.stack.setCurrentIndex(1) # Show loading
        self.loading_page.setText("Generating selected shorts... This may take a while.")
        
        self.worker = Worker("generate_clips", video_path=file_path, clips=selected_clips)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def update_progress(self, message):
        self.loading_page.setText(message)

    def on_processing_finished(self, clips):
        self.results_page.display_clips(clips)
        self.stack.setCurrentIndex(4) # Show results

    def on_processing_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")
        try:
            self.preview_page.set_analyze_busy(False)
            self.preview_page.analysis_card.setVisible(False)
        except AttributeError:
            pass
        self.stack.setCurrentIndex(0) # Go back to input

    def open_settings(self):
        QMessageBox.information(self, "Accounts", "Account settings placeholder.\n\nHere you would connect TikTok, YouTube, and Instagram accounts.")

    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
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
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)
