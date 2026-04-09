"""Backward-compatible exports for camera sensor support."""

from .camera_control.sensor import CameraFrame, G1NavCameraSensor

__all__ = ["CameraFrame", "G1NavCameraSensor"]
