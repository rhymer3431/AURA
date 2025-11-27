import requests
import logging

logger = logging.getLogger(__name__)

class VisClient:
    def __init__(
        self,
        base_url: str = "http://localhost:7000",
        timeout: float = 0.01,
        silent_fail: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.silent_fail = silent_fail

    def _safe_post(self, url, **kwargs):
        try:
            res = requests.post(url, timeout=self.timeout, **kwargs)
            return res.status_code == 200
        except requests.RequestException as e:
            # silent_fail 옵션: 수십 번/초 이상 에러 찍지 않게
            if not self.silent_fail:
                logger.warning(f"POST {url} failed: {e}")
            return False

    def send_frame(self, jpeg_bytes: bytes) -> bool:
        return self._safe_post(
            f"{self.base_url}/frame",
            data=jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )

    def send_graph(self, graph: dict) -> bool:
        return self._safe_post(
            f"{self.base_url}/update",
            json=graph,
        )
