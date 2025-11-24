# Shorty - AI Video Shorts Generator

Shorty is a cross-platform desktop application that automatically turns long YouTube videos into high-quality vertical Shorts/Reels.

## Features

- **Fetch & Download**: Paste a YouTube channel URL to fetch recent videos or drag-and-drop local files.
- **AI Processing**: Automatically detects interesting scenes and converts them to vertical (9:16) format.
- **Quality Rating**: Internal rating system to keep only high-quality clips (>= 8.0).
- **Preview & Edit**: Review generated clips, re-order, delete, or regenerate.
- **Multi-Platform Upload**: One-click upload to TikTok, YouTube Shorts, and Instagram Reels.

## Tech Stack

- **GUI**: Python 3.11+ with PyQt6
- **Video Processing**: moviepy, ffmpeg-python, OpenCV, PySceneDetect
- **Downloading**: yt-dlp
- **Upload Automation**: Playwright, YouTube Data API v3

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python main.py
   ```
