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

import networkx as nx

import networkx as nx
import numpy as np

import networkx as nx
import numpy as np
from math import isnan

import networkx as nx
import numpy as np

# === Scene Graph Layout & Animation State ==================================
_node_positions: dict[str, tuple[float, float]] = {}
_target_positions: dict[str, tuple[float, float]] = {}
_node_ages: dict[str, float] = {}   # 0.0 ~ 1.0 (appear animation)
_edge_ages: dict[tuple[str, str, str], float] = {}
_last_graph_signature: str | None = None
# ============================================================================
def _draw_scene_graph_nx(drawlist, graph, width, height):
    """
    NetworkX 기반 Scene Graph 렌더링
    - Kamada-Kawai 레이아웃 + 첫 노드 중심
    - 레이아웃 transition 애니메이션 (lerp)
    - 노드 최소 거리 강제해서 겹침 완화
    - 노드/엣지 등장 시 fade-in & scale-in
    """
    global _node_positions, _target_positions, _node_ages, _edge_ages, _last_graph_signature

    dpg.delete_item(drawlist, children_only=True)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        dpg.draw_text(
            (20, 20), "No scene graph data.",
            color=(150, 170, 190, 200), size=16,
            parent=drawlist,
        )
        # 상태 초기화
        _node_positions.clear()
        _target_positions.clear()
        _node_ages.clear()
        _edge_ages.clear()
        _last_graph_signature = None
        return

    # ---------------- 그래프 서명 (변화 감지) ----------------
    signature = str(nodes) + str(edges)

    # ---------------- 레이아웃 계산 (그래프 변경 시에만) ----
    if signature != _last_graph_signature:
        G = nx.DiGraph()
        labels = {}

        for n in nodes:
            nid = str(n["id"])
            labels[nid] = n.get("label", "")
            G.add_node(nid)

        for e in edges:
            u = str(e["source"])
            v = str(e["target"])
            G.add_edge(u, v, label=e.get("label", ""))

        anchor = str(nodes[0]["id"])

        if len(nodes) == 1:
            layout = {anchor: (0.0, 0.0)}
        else:
            layout = nx.kamada_kawai_layout(G)

            # anchor를 (0,0)으로 이동시켜 중심 고정
            ax, ay = layout[anchor]
            for k in layout:
                layout[k] = (layout[k][0] - ax, layout[k][1] - ay)

        _target_positions = layout
        _last_graph_signature = signature

        # 새로 등장한 노드/엣지 age 초기화
        new_node_ids = {str(n["id"]) for n in nodes}
        for nid in new_node_ids:
            if nid not in _node_ages:
                _node_ages[nid] = 0.0

        new_edge_keys = {
            (str(e["source"]), str(e["target"]), str(e.get("label", "")))
            for e in edges
        }
        for ek in new_edge_keys:
            if ek not in _edge_ages:
                _edge_ages[ek] = 0.0

        # scene에서 사라진 노드/엣지는 age/state 제거
        _node_positions = {k: v for k, v in _node_positions.items() if k in new_node_ids}
        _node_ages = {k: v for k, v in _node_ages.items() if k in new_node_ids}
        _edge_ages = {k: v for k, v in _edge_ages.items() if k in new_edge_keys}

        # 처음 호출인 경우 현재 위치를 target으로 세팅
        if not _node_positions:
            _node_positions = _target_positions.copy()

    # ---------------- 좌표 보간 (transition animation) ----
    LERP_ALPHA = 0.12  # 0.0~1.0, 클수록 빠르게 목표 위치로 이동

    for nid, tpos in _target_positions.items():
        tx, ty = tpos
        if nid in _node_positions:
            cx, cy = _node_positions[nid]
            _node_positions[nid] = (
                cx * (1.0 - LERP_ALPHA) + tx * LERP_ALPHA,
                cy * (1.0 - LERP_ALPHA) + ty * LERP_ALPHA,
            )
        else:
            _node_positions[nid] = (tx, ty)

    # ---------------- 최소 거리 강제 (노드 겹침 완화) ----
    MIN_DIST = 0.18  # layout 좌표 공간[-1,1] 기준 최소 거리

    nids = list(_node_positions.keys())
    for i in range(len(nids)):
        for j in range(i + 1, len(nids)):
            ni, nj = nids[i], nids[j]
            x1, y1 = _node_positions[ni]
            x2, y2 = _node_positions[nj]
            dx = x2 - x1
            dy = y2 - y1
            dist = float(np.hypot(dx, dy)) + 1e-6
            if dist < MIN_DIST:
                # 서로 반대 방향으로 밀어냄
                push = (MIN_DIST - dist) / 2.0
                ux, uy = dx / dist, dy / dist
                x1 -= ux * push
                y1 -= uy * push
                x2 += ux * push
                y2 += uy * push
                _node_positions[ni] = (x1, y1)
                _node_positions[nj] = (x2, y2)

    # ---------------- 화면 좌표로 스케일링 -------------------
    xs = np.array([p[0] for p in _node_positions.values()])
    ys = np.array([p[1] for p in _node_positions.values()])
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)

    pad = 40
    cx, cy = width / 2.0, height / 2.0
    scale = min(width - pad * 2.0, height - pad * 2.0) / max_range * 0.45

    def map_pos(x: float, y: float) -> tuple[float, float]:
        return cx + x * scale, cy + y * scale

    # ---------------- age 업데이트 (0→1로 상승) ---------------
    AGE_STEP = 0.08  # 프레임당 증가량

    for nid in _node_positions.keys():
        _node_ages[nid] = min(1.0, _node_ages.get(nid, 0.0) + AGE_STEP)

    for e in edges:
        ek = (str(e["source"]), str(e["target"]), str(e.get("label", "")))
        _edge_ages[ek] = min(1.0, _edge_ages.get(ek, 0.0) + AGE_STEP)

    # ---------------- 엣지 그리기 (fade-in) -------------------
    BASE_EDGE_COLOR = (90, 150, 255)  # RGB
    BASE_THICKNESS = 2.0

    for e in edges:
        s = str(e["source"])
        t = str(e["target"])
        lbl = e.get("label", "")
        if s not in _node_positions or t not in _node_positions:
            continue

        x1, y1 = map_pos(*_node_positions[s])
        x2, y2 = map_pos(*_node_positions[t])

        ek = (s, t, str(lbl))
        age = _edge_ages.get(ek, 1.0)
        alpha = int(240 * age)
        thickness = BASE_THICKNESS * (0.5 + 0.5 * age)

        dpg.draw_line(
            (x1, y1),
            (x2, y2),
            color=(BASE_EDGE_COLOR[0], BASE_EDGE_COLOR[1], BASE_EDGE_COLOR[2], alpha),
            thickness=thickness,
            parent=drawlist,
        )

        if lbl:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dpg.draw_text(
                (mx + 6, my + 6),
                lbl[:16],
                size=13,
                color=(210, 230, 255, alpha),
                parent=drawlist,
            )

    # ---------------- 노드 그리기 (scale-in + fade-in) --------
    BASE_RADIUS = 14.0
    OUTLINE_COLOR = (90, 255, 180)
    FILL_COLOR = (40, 200, 130)
    LABEL_COLOR = (235, 245, 255)

    for n in nodes:
        nid = str(n["id"])
        if nid not in _node_positions:
            continue
        x, y = map_pos(*_node_positions[nid])
        raw_label = n.get("label", "")

        age = _node_ages.get(nid, 1.0)
        # ease-out scale
        scale_node = 0.4 + 0.6 * age
        radius = BASE_RADIUS * scale_node
        alpha_circle = int(255 * age)

        dpg.draw_circle(
            (x, y),
            radius,
            color=(OUTLINE_COLOR[0], OUTLINE_COLOR[1], OUTLINE_COLOR[2], alpha_circle),
            fill=(FILL_COLOR[0], FILL_COLOR[1], FILL_COLOR[2], int(230 * age)),
            thickness=2,
            parent=drawlist,
        )

        short = raw_label[:10] + "…" if len(raw_label) > 11 else raw_label
        text = f"{nid}:{short}" if short else nid

        dpg.draw_text(
            (x + radius + 3, y - 7),
            text,
            size=14,
            color=(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2], alpha_circle),
            parent=drawlist,
        )


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

    # 기본 비디오 텍스처 해상도 (레이아웃 기준 사이즈)
    tex_w, tex_h = 640, 480
    with dpg.texture_registry(show=False):
        tex_id = dpg.add_dynamic_texture(
            width=tex_w,
            height=tex_h,
            default_value=[0.0] * (tex_w * tex_h * 4),
        )

    # === 레이아웃 구성 ======================================================
    #
    # main_window
    #   ├─ 상단: 좌측(Video) + 우측(Scene Graph)
    #   └─ 하단: Logs
    #
    with dpg.window(
        label="AURA Monitor",
        tag="main_window",
        width=1600,
        height=900,
        no_title_bar=True,
        no_resize=True,
        no_move=True,
        no_collapse=True,
        no_scrollbar=True,   # 메인 창 스크롤 제거
    ):
        # 상단 메인 영역: 좌우 분할
        with dpg.group(horizontal=True, tag="main_split"):
            # 왼쪽: 비디오 패널
            with dpg.child_window(
                tag="video_panel",
                width=1150,   # 전체 1600 기준 대략 70% 영역
                height=-1,    # 남은 높이 자동
                border=True,
                no_scrollbar=True,   # 내부 스크롤 제거
            ):
                dpg.add_text("Live Video", bullet=False)
                dpg.add_separator()
                image_id = dpg.add_image(tex_id, width=tex_w, height=tex_h)
                fps_text = dpg.add_text("FPS: 0.0")

            # 오른쪽: Scene Graph 패널
            with dpg.child_window(
                tag="graph_panel",
                width=430,    # 우측 약 30% 영역
                height=-1,
                border=True,
                no_scrollbar=True,   # 내부 스크롤 제거
            ):
                dpg.add_text("Scene Graph", bullet=False)
                dpg.add_separator()
                graph_draw = dpg.add_drawlist(
                    tag="graph_draw",
                    width=-1,
                    height=-1,
                )
                dpg.draw_text(
                    (10, 10),
                    "Scene graph will appear here.",
                    parent=graph_draw,
                    color=(180, 200, 220, 220),
                )

        # 하단 로그 영역
        with dpg.child_window(
            tag="log_panel",
            width=-1,
            height=140,
            border=True,
            no_scrollbar=True,   # 로그도 스크롤 없이 고정 영역
        ):
            dpg.add_text("Logs")
            dpg.add_separator()
            log_text = dpg.add_text("Ready.")


    dpg.set_primary_window("main_window", True)
    # =======================================================================

    dpg.setup_dearpygui()
    dpg.show_viewport()

    fps_meter = FpsMeter()

    tex_meta = {"id": tex_id, "w": tex_w, "h": tex_h, "image_id": image_id}

    def ui_loop(sender=None, app_data=None):
        if stop_event.is_set():
            dpg.stop_dearpygui()
            return

        # -------------------- 비디오 업데이트 --------------------
        try:
            frame = ui_bridge.frame_q.get_nowait()
            if not isinstance(frame, np.ndarray):
                return

            frame_lb = _letterbox_to_texture(frame, tex_meta["w"], tex_meta["h"])
            rgba = _ensure_rgba(frame_lb)
            dpg.set_value(tex_meta["id"], rgba.ravel().tolist())

            # video_panel 크기에 맞춰 이미지 크기 재조정
            vw, vh = dpg.get_item_rect_size("video_panel")
            if vw > 0 and vh > 0:
                scale = min(vw / tex_meta["w"], vh / (tex_meta["h"] + 40))  # 상단 텍스트/여백 고려
                new_w = int(tex_meta["w"] * scale)
                new_h = int(tex_meta["h"] * scale)
                dpg.configure_item(tex_meta["image_id"], width=new_w, height=new_h)

            fps = fps_meter.tick()
            dpg.set_value(fps_text, f"FPS: {fps:.1f}")
        except queue.Empty:
            pass

        # -------------------- Scene Graph 업데이트 --------------------
        try:
            graph = ui_bridge.graph_q.get_nowait()
        except queue.Empty:
            graph = None

        gw_panel, gh_panel = dpg.get_item_rect_size("graph_panel")
        if gw_panel > 0 and gh_panel > 0:
            # 텍스트/여백 감안해서 drawlist 사이즈 조정
            dpg.configure_item("graph_draw", width=gw_panel - 20, height=gh_panel - 40)

        if graph:
            _draw_scene_graph_nx("graph_draw", graph, gw_panel - 20, gh_panel - 40)

    while dpg.is_dearpygui_running():
        ui_loop()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
