from PyQt6.QtCore import QThread, pyqtSignal
from src.core.downloader import VideoDownloader
from src.core.processor import VideoProcessor
import os
import re

class Worker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    analysis_progress = pyqtSignal(int, int)

    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        self.downloader = VideoDownloader()
        self.processor = VideoProcessor()

    def run(self):
        try:
            if self.task_type == "download_channel":
                url = self.kwargs.get("url")
                self.progress.emit("Fetching channel videos...")
                videos = self.downloader.get_channel_videos(url)
                self.finished.emit(videos)
                
            elif self.task_type == "download_video":
                url = self.kwargs.get("url")
                self.progress.emit(f"Downloading video: {url}")
                def hook(d):
                    if d['status'] == 'downloading':
                        p = d.get('_percent_str', '0%')
                        # Remove ANSI escape codes
                        p = re.sub(r'\x1b\[[0-9;]*m', '', p).replace('%','')
                        self.progress.emit(f"Downloading: {p}%")
                video_path = self.downloader.download_video(url, progress_hook=hook)
                self.finished.emit(video_path)

            elif self.task_type == "analyze_file":
                video_path = self.kwargs.get("video_path")
                self.progress.emit("Analyzing video for shorts candidates...")

                def progress_cb(current, total):
                    self.analysis_progress.emit(current, total)
                    if total:
                        percent = int((current / total) * 100)
                        self.progress.emit(f"Analyzing video... {percent}% ({current}/{total})")

                clips = self.processor.analyze_video(video_path, progress_callback=progress_cb)
                self.finished.emit(clips)

            elif self.task_type == "generate_clips":
                video_path = self.kwargs.get("video_path")
                clips = self.kwargs.get("clips", [])
                total = len(clips)
                generated_clips = []
                base_name = os.path.basename(video_path)
                for idx, clip in enumerate(clips, start=1):
                    score = clip.get("score")
                    label = f"Score: {score}" if score is not None else ""
                    # Emit progress as percentage
                    percent = int((idx - 1) / total * 100) if total > 0 else 0
                    self.progress.emit(f"GENERATION_PROGRESS:{percent}")
                    start = clip.get("start")
                    end = clip.get("end")
                    if start is None or end is None:
                        continue
                    clip_id = clip.get("id") or f"clip_{clip.get('index', idx-1)}"
                    output_filename = clip.get("output_filename") or f"{clip_id}_{base_name}"
                    clip_path = self.processor.create_vertical_short(video_path, start, end, output_filename)
                    generated_clips.append(clip_path)
                    # Emit 100% when done with this clip
                    percent = int(idx / total * 100) if total > 0 else 100
                    self.progress.emit(f"GENERATION_PROGRESS:{percent}")
                self.finished.emit(generated_clips)

            elif self.task_type == "process_file":
                video_path = self.kwargs.get("video_path")
                self.progress.emit("Detecting scenes...")
                clips = self.processor.analyze_video(video_path)
                selected = [clip for clip in clips if clip.get("score", 0) >= 8.0]
                generated_clips = self.processor.generate_clips(video_path, selected)
                self.finished.emit(generated_clips)
                
        except Exception as e:
            self.error.emit(str(e))
