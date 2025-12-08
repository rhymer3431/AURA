"""Offline demo runner for the perception pipeline."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from application.demo.offline_demo import run_video_demo


def main():
    run_video_demo(video_path="input/video.mp4", enable_logging=False)


if __name__ == "__main__":
    main()
