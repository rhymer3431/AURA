from typing import Any, Dict, List
import asyncio
import json

import uvicorn
import cv2
import numpy as np
from av import VideoFrame
from fastapi import FastAPI, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FrameBufferTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def recv(self) -> VideoFrame:
        frame = await self.queue.get()
        return frame

    def push(self, jpeg_bytes: bytes):
        # Decode JPEG bytes to numpy array then to VideoFrame
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        # drop old frame if queue full
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass


video_track = FrameBufferTrack()
data_channels = []


async def create_peer_connection(offer: Dict[str, Any]):
    pc = RTCPeerConnection()

    @pc.on("datachannel")
    def on_datachannel(channel):
        data_channels.append(channel)

    pc.addTrack(video_track)

    remote_desc = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    await pc.setRemoteDescription(remote_desc)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    pc = await create_peer_connection(params)
    return JSONResponse(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


@app.post("/frame")
async def upload_frame(request: Request):
    jpeg_bytes = await request.body()
    video_track.push(jpeg_bytes)
    return JSONResponse({"status": "ok", "size": len(jpeg_bytes)})


@app.post("/update")
async def update_graph(graph: Dict[str, Any]):
    payload = json.dumps(graph)
    stale = []
    for ch in data_channels:
        try:
            ch.send(payload)
        except WebSocketDisconnect:
            stale.append(ch)
        except Exception:
            stale.append(ch)
    for ch in stale:
        if ch in data_channels:
            data_channels.remove(ch)
    return JSONResponse({"status": "ok", "clients": len(data_channels)})


# Optional sample for quick check
sample_graph = {
    "nodes": [
        {"id": "1", "label": "robot"},
        {"id": "2", "label": "person"},
    ],
    "edges": [{"source": "1", "target": "2", "label": "approaching"}],
}


@app.post("/update_sample")
async def update_sample():
    return await update_graph(sample_graph)


if __name__ == "__main__":
    uvicorn.run("server.webrtc_server:app", host="0.0.0.0", port=7000, reload=False)
