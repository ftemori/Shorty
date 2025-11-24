from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal
from src.ui.player_widget import VideoPlayer
import os

class VideoPreviewWidget(QWidget):
    analyze_clicked = pyqtSignal(str) # Emits file path
    generate_requested = pyqtSignal(str, list) # Emits file path + selected clips
    back_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.analyzed_clips = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Container for video display area
        self.video_container = QWidget()
        video_container_layout = QVBoxLayout(self.video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.setSpacing(0)
        
        # Video Info Frame - shown by default when no video
        self.video_info_frame = QFrame()
        self.video_info_frame.setFixedSize(900, 505)
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
        self.player.setFixedSize(900, 505)
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
        
        # Separator Line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.separator.setStyleSheet("background-color: #444; margin-top: 50px;")
        self.separator.setFixedHeight(2)
        self.separator.setVisible(False)
        layout.addWidget(self.separator)

        # Analysis Results Card
        self.analysis_card = QFrame()
        self.analysis_card.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 12px;
                border: 1px solid #3a3a3a;
            }
        """)
        self.analysis_card_layout = QVBoxLayout(self.analysis_card)
        self.analysis_card_layout.setSpacing(12)

        analysis_title = QLabel("Select which clips to generate")
        analysis_title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        self.analysis_card_layout.addWidget(analysis_title)

        self.analysis_status = QLabel("Run an analysis to preview clip ratings.")
        self.analysis_status.setStyleSheet("color: #aaa;")
        self.analysis_card_layout.addWidget(self.analysis_status)

        self.analysis_progress_label = QLabel("")
        self.analysis_progress_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self.analysis_card_layout.addWidget(self.analysis_progress_label)

        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setRange(0, 100)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_bar.setTextVisible(False)
        self.analysis_card_layout.addWidget(self.analysis_progress_bar)

        self.analysis_list = QListWidget()
        self.analysis_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        self.analysis_list.itemChanged.connect(self.on_selection_changed)
        self.analysis_card_layout.addWidget(self.analysis_list)

        self.generate_btn = QPushButton("⚡ Generate Selected Shorts")
        self.generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.generate_btn.setStyleSheet("""
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
                background-color: #1d4f34;
                color: #888;
            }
        """)
        self.generate_btn.clicked.connect(self.on_generate_selected)
        self.analysis_card_layout.addWidget(self.generate_btn)

        layout.addWidget(self.analysis_card)
        self.analysis_card.setVisible(False)

    def set_video(self, path):
        self.video_path = path
        filename = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        self.info_label.setText(f"File: {filename}\nSize: {size_mb:.1f} MB")
        
        # Hide info frame and show video player
        self.video_info_frame.hide()
        self.player.show()
        self.player.set_source(path)
        
        self.reset_analysis()

    def reset_analysis(self):
        self.analyzed_clips = []
        # Don't show/hide player or info frame here - that's handled by set_video
        self.separator.setVisible(False)
        self.analysis_progress_label.setVisible(False)
        self.analysis_progress_bar.setVisible(False)
        self.analysis_progress_bar.setValue(0)
        self.analysis_list.blockSignals(True)
        self.analysis_list.clear()
        self.analysis_list.blockSignals(False)
        self.analysis_status.setText("Run an analysis to preview clip ratings.")
        self.analysis_card.setVisible(False)
        self.generate_btn.setEnabled(False)
        self.set_analyze_busy(False)

    def begin_analysis(self):
        # Hide player during analysis
        self.player.setVisible(False)
        self.separator.setVisible(True)
        self.analysis_card.setVisible(True)
        self.analysis_status.setText("Analyzing video... This may take a moment.")
        self.analysis_progress_label.setText("Preparing...")
        self.analysis_progress_label.setVisible(True)
        self.analysis_progress_bar.setVisible(True)
        self.analysis_progress_bar.setRange(0, 0) # Indeterminate until we know total
        self.analysis_list.blockSignals(True)
        self.analysis_list.clear()
        self.analysis_list.blockSignals(False)
        self.analysis_list.setEnabled(False)
        self.generate_btn.setEnabled(False)

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
        # Keep player hidden during analysis results
        self.player.stop()
        self.player.setVisible(False)
        self.analysis_progress_label.setVisible(False)
        self.analysis_progress_bar.setVisible(False)

        self.analyzed_clips = clips or []
        self.analysis_list.blockSignals(True)
        self.analysis_list.clear()

        if not self.analyzed_clips:
            self.analysis_status.setText("No eligible clips were found between 15s and 60s.")
            self.analysis_list.setEnabled(False)
        else:
            self.analysis_list.setEnabled(True)
            for clip in self.analyzed_clips:
                index = clip.get("index", 0) + 1
                score = clip.get("score", 0)
                start = self.format_timestamp(clip.get("start", 0))
                end = self.format_timestamp(clip.get("end", 0))
                duration = clip.get("duration", clip.get("end", 0) - clip.get("start", 0))
                item_text = f"Clip {index} • Score {score} • {start} - {end} ({duration:.1f}s)"
                item = QListWidgetItem(item_text)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                default_checked = score >= 8.0
                item.setCheckState(Qt.CheckState.Checked if default_checked else Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, clip)
                self.analysis_list.addItem(item)
            self.analysis_status.setText("Select the clips you want to generate.")

        self.analysis_list.blockSignals(False)
        self.analysis_card.setVisible(True)
        self.set_analyze_busy(False)
        self.update_generate_state()

    def get_selected_clips(self):
        selected = []
        for i in range(self.analysis_list.count()):
            item = self.analysis_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                clip = item.data(Qt.ItemDataRole.UserRole)
                if clip:
                    selected.append(clip)
        return selected

    def update_generate_state(self):
        any_selected = any(
            self.analysis_list.item(i).checkState() == Qt.CheckState.Checked
            for i in range(self.analysis_list.count())
        )
        self.generate_btn.setEnabled(any_selected)

    def on_selection_changed(self, _item):
        self.update_generate_state()

    def on_analyze(self):
        # Stop player before analysis
        if hasattr(self, 'player'):
            self.player.stop()
        if self.video_path:
            self.begin_analysis()
            self.set_analyze_busy(True)
            self.analyze_clicked.emit(self.video_path)

    def on_generate_selected(self):
        # Stop player before generation
        if hasattr(self, 'player'):
            self.player.stop()
        if not self.video_path:
            return
        selected = self.get_selected_clips()
        if selected:
            self.generate_requested.emit(self.video_path, selected)

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
