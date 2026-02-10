"""Offline demo runner for the perception pipeline."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.application.demo.offline_demo import run_video_demo

video_path="backend/input/video.mp4"
def main():
    run_video_demo(video_path=video_path, enable_logging=True)


if __name__ == "__main__":
    main()
