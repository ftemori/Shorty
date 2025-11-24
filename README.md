# Shorty - Video Shorts Generator

Shorty is a cross-platform desktop application that automatically turns long YouTube videos into high-quality vertical Shorts/Reels using **industry-standard AI algorithms**.

## Features

- **Fetch & Download**: Paste a YouTube channel URL to fetch recent videos or drag-and-drop local files.
- **E²SAVS Algorithm**: State-of-the-art viral clip extraction with audio excitement analysis, visual saliency detection, and intelligent boundary detection.
- **AFAPZ Face Tracking**: Cinematic face-anchored pan-zoom (same technology as TikTok, Adobe Premiere, CapCut) with:
  - MediaPipe face detection + KLT tracking
  - Smooth Kalman filtering
  - TikTok-style positioning (27% from top)
  - Adaptive zoom (1.0→1.15x with snap zoom)
- **Quality Rating**: Internal ViralScore system (1.0-10.0) keeps only high-quality clips.
- **Smart Boundaries**: Clips end at natural sentence/topic boundaries (no mid-sentence cuts).
- **Preview & Edit**: Review generated clips, re-order, delete, or regenerate.
- **Multi-Platform Upload**: One-click upload to TikTok, YouTube Shorts, and Instagram Reels.

## Tech Stack

- **GUI**: Python 3.11+ with PyQt6
- **Video Processing**: moviepy, ffmpeg-python, OpenCV, PySceneDetect, MediaPipe
- **Audio Analysis**: librosa (RMS, spectral flux, crest factor)
- **Face Tracking**: MediaPipe Face Detection, KLT optical flow
- **Smoothing**: Kalman filters, Savitzky-Golay filters (scipy)
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
