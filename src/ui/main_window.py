from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPalette, QColor, QIcon
from src.ui.input_widget import InputWidget
from src.ui.results_widget import ResultsWidget
from src.ui.selection_widget import VideoSelectionWidget
from src.ui.preview_widget import VideoPreviewWidget
from src.core.worker import Worker
import os

class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.start = QPoint(0, 0)
        self.pressing = False
        self.init_ui()

    def init_ui(self):
        self.setFixedHeight(30)
        self.setStyleSheet("background-color: #202020; border-bottom: 1px solid #333;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)

        # Title
        title_label = QLabel("Shorty")
        title_label.setStyleSheet("font-weight: bold; color: white; font-size: 14px; padding-left: 10px;")
        layout.addWidget(title_label)

        layout.addStretch()

        # Window Controls
        self.min_btn = self.create_btn("-", self.parent.showMinimized)
        self.max_btn = self.create_btn("□", self.toggle_max)
        self.close_btn = self.create_btn("✕", self.parent.close, is_close=True)

        layout.addWidget(self.min_btn)
        layout.addWidget(self.max_btn)
        layout.addWidget(self.close_btn)

    def create_btn(self, text, callback, is_close=False):
        btn = QPushButton(text)
        btn.setFixedSize(45, 30)
        btn.clicked.connect(callback)
        bg_hover = "#c42b1c" if is_close else "#444"
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: #aaa;
                border: none;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {bg_hover};
                color: white;
            }}
        """)
        return btn

    def toggle_max(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

    def mousePressEvent(self, event):
        self.start = self.mapToGlobal(event.pos())
        self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing and not self.parent.isMaximized():
            end = self.mapToGlobal(event.pos())
            movement = end - self.start
            self.parent.setGeometry(self.parent.x() + movement.x(),
                                  self.parent.y() + movement.y(),
                                  self.parent.width(),
                                  self.parent.height())
            self.start = end

    def mouseReleaseEvent(self, event):
        self.pressing = False

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
        
        # Content Layout
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(20)
        self.main_layout.addLayout(self.content_layout)

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
        top_section.setFixedHeight(300) 
        self.content_layout.addWidget(top_section)

        # 3. Middle Section: Video Player (Centered, 60% width)
        # We use a container to center it
        player_container = QWidget()
        player_layout = QHBoxLayout(player_container)
        player_layout.setContentsMargins(0, 50, 0, 0) # 50px margin top
        player_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.preview_widget = VideoPreviewWidget()
        
        # Connect signals
        self.preview_widget.analyze_clicked.connect(self.start_analysis)
        self.preview_widget.generate_requested.connect(self.start_generation)
        # back_clicked is probably not needed anymore as we are single page, 
        # but maybe it resets the player?
        self.preview_widget.back_clicked.connect(self.reset_player)

        player_layout.addWidget(self.preview_widget)
        self.content_layout.addWidget(player_container)
        
        # 4. Bottom Section: Accounts Button
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        self.accounts_btn = QPushButton("Accounts")
        self.accounts_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.accounts_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #444;
            }
        """)
        self.accounts_btn.clicked.connect(self.open_settings)
        bottom_layout.addWidget(self.accounts_btn)
        
        self.content_layout.addLayout(bottom_layout)

        # Results Overlay (Hidden by default, or we can use a dialog)
        self.results_widget = ResultsWidget()
        self.results_widget.hide() 
        # For now, we might need to show results in a separate window or replace content.
        # Given "all on same page", maybe replace the player area?
        


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
        
        self.worker = Worker("download_video", url=url)
        self.worker.finished.connect(self.preview_widget.set_video)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def start_analysis(self, file_path):
        self.worker = Worker("analyze_file", video_path=file_path)
        self.worker.analysis_progress.connect(self.preview_widget.update_analysis_progress)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()

    def on_analysis_finished(self, clips):
        self.preview_widget.show_analysis(clips)

    def start_generation(self, file_path, selected_clips):
        # Show generating state
        self.preview_widget.analysis_status.setText("Generating shorts... Please wait.")
        self.preview_widget.generate_btn.setEnabled(False)
        
        self.worker = Worker("generate_clips", video_path=file_path, clips=selected_clips)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()
        
    def on_generation_finished(self, clips):
        # Show results
        # Since we want everything on one page, maybe we open the results folder 
        # or show a "Done" message and list the files in the preview area?
        # The ResultsWidget logic is: display clips and allow open folder.
        # Let's show a message box for now or use the ResultsWidget as a popup.
        self.results_widget.display_clips(clips)
        self.results_widget.show()
        self.results_widget.setWindowTitle("Generated Shorts")
        self.results_widget.resize(600, 400)
        
        self.preview_widget.analysis_status.setText("Generation complete!")
        self.preview_widget.generate_btn.setEnabled(True)

    def reset_player(self):
        # Reset logic if needed
        pass

    def on_processing_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")
        self.preview_widget.set_analyze_busy(False)

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
