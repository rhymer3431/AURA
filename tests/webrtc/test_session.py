from __future__ import annotations

import asyncio
from types import SimpleNamespace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from webrtc.config import WebRTCGatewayConfig
from webrtc.session import PeerSessionManager


class _FakeSession:
    def __init__(self, session_id: str, config: WebRTCGatewayConfig, subscriber) -> None:  # noqa: ANN001
        self.session_id = session_id
        self.config = config
        self.subscriber = subscriber
        self.track_roles = ["rgb"]
        self.closed = False
        self.accepted: list[dict[str, object]] = []

    async def accept_offer(self, offer_payload: dict[str, object]):
        self.accepted.append(dict(offer_payload))
        return SimpleNamespace(sdp=f"answer-{self.session_id}", type="answer")

    async def close(self) -> None:
        self.closed = True


def test_peer_session_manager_replaces_previous_session() -> None:
    async def scenario() -> None:
        created: list[_FakeSession] = []

        def factory(session_id: str, config: WebRTCGatewayConfig, subscriber) -> _FakeSession:  # noqa: ANN001
            session = _FakeSession(session_id, config, subscriber)
            created.append(session)
            return session

        manager = PeerSessionManager(WebRTCGatewayConfig(), subscriber=object(), session_factory=factory)
        first, first_answer = await manager.accept_offer({"sdp": "offer-1", "type": "offer"})
        second, second_answer = await manager.accept_offer({"sdp": "offer-2", "type": "offer"})

        assert first_answer.type == "answer"
        assert second_answer.sdp.startswith("answer-")
        assert created[0].accepted == [{"sdp": "offer-1", "type": "offer"}]
        assert created[0].closed is True
        assert manager.active_session is second

        await manager.close()
        assert created[1].closed is True

    asyncio.run(scenario())
