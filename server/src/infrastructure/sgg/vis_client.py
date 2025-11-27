import requests


class VisClient:
    def __init__(self, base_url: str = "http://localhost:7000"):
        self.base_url = base_url.rstrip("/")

    def send_frame(self, jpeg_bytes: bytes):
        url = f"{self.base_url}/frame"
        headers = {"Content-Type": "image/jpeg"}
        requests.post(url, data=jpeg_bytes, headers=headers, timeout=0.5)

    def send_graph(self, graph: dict):
        url = f"{self.base_url}/update"
        requests.post(url, json=graph, timeout=0.5)
