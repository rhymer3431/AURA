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


def _letterbox_to_texture(frame_bgr: np.ndarray, tex_w: int, tex_h: int) -> np.ndarray:
    """
    Fit frame into texture size while preserving aspect ratio (letterbox).
    Returns a BGR image of shape (tex_h, tex_w, 3).
    """
    fh, fw, _ = frame_bgr.shape
    scale = min(tex_w / fw, tex_h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h))
    canvas = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)
    x0 = (tex_w - new_w) // 2
    y0 = (tex_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def run_dpg(ui_bridge, stop_event: threading.Event, target_fps: int = 60):
    """
    Starts the DearPyGui UI loop. Should be called from main thread.
    ui_bridge: provides frame_q and graph_q queues.
    stop_event: set from outside to request shutdown.
    target_fps: UI refresh target.
    """
    dpg.create_context()
    dpg.create_viewport(title="AURA Monitor (DearPyGui)", width=1600, height=900)

    tex_w, tex_h = 640, 480  # base texture size (letterboxed, keeps aspect)
    with dpg.texture_registry(show=False) as tex_reg:
        tex_id = dpg.add_dynamic_texture(
            width=tex_w,
            height=tex_h,
            default_value=[0.0] * (tex_w * tex_h * 4),
        )

    with dpg.window(label="AURA Monitor", tag="main_window", width=1600, height=900, no_title_bar=True):
        with dpg.group(horizontal=True):
            with dpg.child_window(label="Video", tag="video_panel", width=-1, height=-1):
                image_id = dpg.add_image(tex_id, width=tex_w, height=tex_h)
                fps_text = dpg.add_text("FPS: 0.0")
            with dpg.child_window(label="Scene Graph", tag="graph_panel", width=420, height=-1):
                graph_draw = dpg.add_drawlist(tag="graph_draw", width=-1, height=-1)
                dpg.draw_text((10, 10), "Scene graph will appear here.", parent=graph_draw, color=(180, 200, 220, 220))
        with dpg.child_window(label="Logs", width=-1, height=120):
            log_text = dpg.add_text("Ready.")

    dpg.set_primary_window("main_window", True)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    fps_meter = FpsMeter()

    tex_meta = {"id": tex_id, "w": tex_w, "h": tex_h, "image_id": image_id}

    def ui_loop(sender=None, app_data=None):
        if stop_event.is_set():
            dpg.stop_dearpygui()
            return
        # frame update
        try:
            frame = ui_bridge.frame_q.get_nowait()
            # ensure numpy array
            if not isinstance(frame, np.ndarray):
                return
            # letterbox to texture size to preserve aspect ratio
            frame_lb = _letterbox_to_texture(frame, tex_meta["w"], tex_meta["h"])
            rgba = _ensure_rgba(frame_lb)
            dpg.set_value(tex_meta["id"], rgba.ravel().tolist())
            # scale image widget to fit panel while preserving aspect
            vw, vh = dpg.get_item_rect_size("video_panel")
            if vw > 0 and vh > 0:
                scale = min(vw / tex_meta["w"], vh / tex_meta["h"])
                new_w = int(tex_meta["w"] * scale)
                new_h = int(tex_meta["h"] * scale)
                dpg.configure_item(tex_meta["image_id"], width=new_w, height=new_h)
            fps = fps_meter.tick()
            dpg.set_value(fps_text, f"FPS: {fps:.1f}")
        except queue.Empty:
            pass
        # graph update
        try:
            graph = ui_bridge.graph_q.get_nowait()
        except queue.Empty:
            graph = None
        # Resize drawlist to panel each frame to ensure it's visible
        gw_panel, gh_panel = dpg.get_item_rect_size("graph_panel")
        if gw_panel > 0 and gh_panel > 0:
            dpg.configure_item("graph_draw", width=gw_panel - 10, height=gh_panel - 10)
        if graph:
            _draw_scene_graph("graph_draw", graph, gw_panel, gh_panel)

    # Manual render loop (compatible with older DearPyGui APIs)
    while dpg.is_dearpygui_running():
        ui_loop()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
