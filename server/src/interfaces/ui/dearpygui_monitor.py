"""
DearPyGui-based local monitor for live video and scene graph visualization.
Runs on the main thread; the perception pipeline should run on a worker thread.
"""

import threading
import time
import queue
import numpy as np
import cv2
from dearpygui import dearpygui as dpg


class FpsMeter:
    """Simple EMA FPS meter for UI refresh rate."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.last = None
        self.fps = 0.0

    def tick(self):
        now = time.time()
        if self.last is not None:
            dt = now - self.last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = self.alpha * self.fps + (1 - self.alpha) * inst
        self.last = now
        return self.fps


def _ensure_rgba(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR np.ndarray to float32 RGBA in [0,1] for DearPyGui dynamic texture."""
    rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    return rgba.astype(np.float32) / 255.0


def _draw_scene_graph(drawlist, graph, width, height):
    """
    Very lightweight scene-graph renderer on a drawlist.
    Nodes placed on a circle; edges with labels at midpoints.
    """
    dpg.delete_item(drawlist, children_only=True)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    n = max(len(nodes), 1)
    radius = min(width, height) * 0.35
    cx, cy = width * 0.5, height * 0.5
    positions = {}
    for i, node in enumerate(nodes):
        theta = 2 * np.pi * i / n
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        positions[str(node["id"])] = (x, y)
    # edges
    for e in edges:
        s, t = str(e["source"]), str(e["target"])
        if s in positions and t in positions:
            dpg.draw_line(positions[s], positions[t], color=(59, 123, 255, 200), parent=drawlist, thickness=2)
            mx = (positions[s][0] + positions[t][0]) / 2
            my = (positions[s][1] + positions[t][1]) / 2
            dpg.draw_text((mx + 4, my + 4), e.get("label", ""), color=(157, 196, 255, 255), size=14, parent=drawlist)
    # nodes
    for node in nodes:
        nid = str(node["id"])
        if nid not in positions:
            continue
        x, y = positions[nid]
        dpg.draw_circle((x, y), 10, color=(58, 255, 20, 255), fill=(58, 255, 20, 80), thickness=2, parent=drawlist)
        lbl = f'{nid}:{node.get("label", "")}'
        dpg.draw_text((x + 12, y - 6), lbl, color=(224, 247, 255, 255), size=14, parent=drawlist)


def run_dpg(ui_bridge, stop_event: threading.Event, target_fps: int = 60):
    """
    Starts the DearPyGui UI loop. Should be called from main thread.
    ui_bridge: provides frame_q and graph_q queues.
    stop_event: set from outside to request shutdown.
    target_fps: UI refresh target.
    """
    dpg.create_context()
    dpg.create_viewport(title="AURA Monitor (DearPyGui)", width=1600, height=900)

    with dpg.texture_registry(show=False):
        tex_id = dpg.add_dynamic_texture(width=640, height=480, default_value=np.zeros((480, 640, 4), dtype=np.float32))

    with dpg.window(label="AURA Monitor", width=1600, height=900):
        with dpg.group(horizontal=True):
            with dpg.child_window(label="Video", width=1050, height=720):
                dpg.add_image(tex_id)
                fps_text = dpg.add_text("FPS: 0.0")
            with dpg.child_window(label="Scene Graph", width=500, height=720):
                graph_draw = dpg.add_drawlist(width=500, height=700)
        with dpg.child_window(label="Logs", width=-1, height=120):
            log_text = dpg.add_text("Ready.")

    dpg.setup_dearpygui()
    dpg.show_viewport()

    fps_meter = FpsMeter()

    def ui_loop(sender=None, app_data=None):
        if stop_event.is_set():
            dpg.stop_dearpygui()
            return
        # frame update
        try:
            frame = ui_bridge.frame_q.get_nowait()
            rgba = _ensure_rgba(frame)
            h, w, _ = rgba.shape
            dpg.configure_item(tex_id, width=w, height=h)
            dpg.set_value(tex_id, rgba.flatten())
            fps = fps_meter.tick()
            dpg.set_value(fps_text, f"FPS: {fps:.1f}")
        except queue.Empty:
            pass
        # graph update
        try:
            graph = ui_bridge.graph_q.get_nowait()
        except queue.Empty:
            graph = None
        if graph:
            gw = dpg.get_item_width(graph_draw)
            gh = dpg.get_item_height(graph_draw)
            _draw_scene_graph(graph_draw, graph, gw, gh)
        # reschedule next frame callback
        dpg.set_frame_callback(0, ui_loop)

    # kick off the first frame callback
    dpg.set_frame_callback(0, ui_loop)

    dpg.start_dearpygui()
    dpg.destroy_context()
