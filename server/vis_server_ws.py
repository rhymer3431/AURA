from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

# FastAPI app with permissive CORS for local dev
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Latest graph cache (shared across connections)
latest_graph: Dict[str, Any] = {
    "nodes": [{"id": "1", "label": "person"}],
    "edges": [{"source": "1", "target": "2", "label": "beside"}],
}
# Latest JPEG frame buffer (bytes)
latest_frame: bytes | None = None


class ConnectionManager:
    """Manages active WebSocket connections and broadcast."""

    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)
        # send the current graph state immediately
        await websocket.send_json(latest_graph)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: Any):
        stale = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except WebSocketDisconnect:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # keep the connection alive; client messages are ignored
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/update")
async def update_graph(graph: Dict[str, Any]):
    """Update the latest graph and broadcast to all clients."""
    global latest_graph
    latest_graph = graph
    await manager.broadcast(latest_graph)
    return JSONResponse({"status": "ok", "clients": len(manager.active)})


@app.post("/frame")
async def upload_frame(request: Request):
    """
    Accept raw JPEG bytes (Content-Type: image/jpeg) and cache as latest_frame.
    Intended to be called by the robotics pipeline per-frame.
    """
    global latest_frame
    latest_frame = await request.body()
    return JSONResponse({"status": "ok", "size": len(latest_frame)})


async def mjpeg_stream():
    """
    Simple MJPEG streaming generator using the latest cached frame.
    If no frame yet, waits briefly.
    """
    boundary = "frame"
    while True:
        if latest_frame:
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(latest_frame)).encode() + b"\r\n\r\n"
            )
            yield latest_frame
            yield b"\r\n"
        await asyncio.sleep(0.05)


@app.get("/video")
async def video_feed():
    """
    MJPEG video feed at http://localhost:7000/video
    """
    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


# Convenience sample graph for quick manual testing
sample_graph = {
    "nodes": [
        {"id": "1", "label": "robot"},
        {"id": "2", "label": "person"},
        {"id": "3", "label": "box"},
    ],
    "edges": [
        {"source": "1", "target": "2", "label": "approaching"},
        {"source": "2", "target": "3", "label": "beside"},
    ],
}


@app.post("/update_sample")
async def update_sample():
    return await update_graph(sample_graph)


if __name__ == "__main__":
    uvicorn.run("vis_server_ws:app", host="0.0.0.0", port=7000, reload=False)
