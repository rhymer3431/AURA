from __future__ import annotations

import base64
import copy
import io
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import requests

from common.cv2_compat import cv2
from memory.models import MemoryContextBundle

DecisionMode = Literal["pixel_goal", "stop", "yaw_left", "yaw_right", "look_down", "wait"]

_TURN_LEFT_TOKENS = ("←", "turn left", "left")
_TURN_RIGHT_TOKENS = ("→", "turn right", "right")
_LOOK_DOWN_TOKENS = ("↓", "lookdown", "look down", "tilt down")


def build_vlm_endpoint(url: str) -> str:
    raw = str(url).strip()
    if raw == "":
        return "http://127.0.0.1:8080/v1/chat/completions"
    if raw.endswith("/v1/chat/completions"):
        return raw
    return raw.rstrip("/") + "/v1/chat/completions"


def extract_chat_content(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise ValueError("VLM response missing choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
        return "\n".join(parts)
    raise ValueError("Unsupported VLM content format.")


@dataclass(frozen=True)
class System2Decision:
    mode: DecisionMode
    pixel_goal: tuple[int, int] | None = None
    reason: str = ""
    raw_text: str = ""
    history_frame_ids: tuple[int, ...] = ()
    needs_requery: bool = False


@dataclass(frozen=True)
class System2Request:
    frame_id: int
    width: int
    height: int
    history_frame_ids: tuple[int, ...]
    body: dict[str, Any]


@dataclass(frozen=True)
class System2SessionResult:
    ok: bool
    decision: System2Decision | None = None
    latency_ms: float = 0.0
    error: str = ""
    source: str = "llm"


@dataclass(frozen=True)
class System2SessionConfig:
    endpoint: str
    model: str
    temperature: float = 0.2
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    timeout_sec: float = 35.0
    num_history: int = 8
    max_images_per_request: int = 3
    mode: Literal["llm", "mock"] = "llm"


@dataclass(frozen=True)
class _ObservedFrame:
    frame_id: int
    image_bgr: np.ndarray


def parse_system2_output(
    raw_text: str,
    *,
    width: int,
    height: int,
    history_frame_ids: tuple[int, ...] = (),
) -> System2Decision:
    text = str(raw_text).strip()
    lowered = text.lower()
    numbers = [int(token) for token in re.findall(r"-?\d+", text)]
    if len(numbers) >= 2:
        # Official InternVLA parsing treats the textual order as (y, x).
        pixel_y = int(np.clip(numbers[0], 0, max(int(height) - 1, 0)))
        pixel_x = int(np.clip(numbers[1], 0, max(int(width) - 1, 0)))
        return System2Decision(
            mode="pixel_goal",
            pixel_goal=(pixel_x, pixel_y),
            reason=text or "pixel_goal",
            raw_text=text,
            history_frame_ids=tuple(history_frame_ids),
            needs_requery=False,
        )

    if re.search(r"\bstop\b", lowered):
        return System2Decision(
            mode="stop",
            reason=text or "STOP",
            raw_text=text,
            history_frame_ids=tuple(history_frame_ids),
            needs_requery=False,
        )

    left_index = _first_token_index(lowered, _TURN_LEFT_TOKENS)
    right_index = _first_token_index(lowered, _TURN_RIGHT_TOKENS)
    if left_index >= 0 and (right_index < 0 or left_index <= right_index):
        return System2Decision(
            mode="yaw_left",
            reason=text or "LEFT",
            raw_text=text,
            history_frame_ids=tuple(history_frame_ids),
            needs_requery=False,
        )
    if right_index >= 0:
        return System2Decision(
            mode="yaw_right",
            reason=text or "RIGHT",
            raw_text=text,
            history_frame_ids=tuple(history_frame_ids),
            needs_requery=False,
        )

    if _first_token_index(lowered, _LOOK_DOWN_TOKENS) >= 0:
        return System2Decision(
            mode="look_down",
            reason=text or "LOOK_DOWN",
            raw_text=text,
            history_frame_ids=tuple(history_frame_ids),
            needs_requery=True,
        )

    return System2Decision(
        mode="wait",
        reason=text or "unparsed_system2_output",
        raw_text=text,
        history_frame_ids=tuple(history_frame_ids),
        needs_requery=True,
    )


class System2Session:
    def __init__(self, config: System2SessionConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._instruction = ""
        self._frames: list[_ObservedFrame] = []
        self._last_output = ""
        self._last_reason = ""
        self._last_history_frame_ids: tuple[int, ...] = ()
        self._last_decision_mode: DecisionMode = "wait"
        self._last_needs_requery = False

    def reset(self, instruction: str) -> None:
        with self._lock:
            self._instruction = str(instruction).strip()
            self._frames = []
            self._last_output = ""
            self._last_reason = ""
            self._last_history_frame_ids = ()
            self._last_decision_mode = "wait"
            self._last_needs_requery = False

    def observe(self, frame_id: int, image_bgr: np.ndarray) -> None:
        image = np.asarray(image_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"System2Session.observe expects [H,W,3] BGR input, got {image.shape}")
        with self._lock:
            if self._frames and self._frames[-1].frame_id == int(frame_id):
                self._frames[-1] = _ObservedFrame(frame_id=int(frame_id), image_bgr=image.copy())
            else:
                self._frames.append(_ObservedFrame(frame_id=int(frame_id), image_bgr=image.copy()))

    def prepare_request(
        self,
        events: dict[str, Any] | None = None,
        memory_context: MemoryContextBundle | None = None,
    ) -> System2Request:
        with self._lock:
            if self._instruction == "":
                raise RuntimeError("System2Session.reset() must be called with a non-empty instruction first.")
            if not self._frames:
                raise RuntimeError("System2Session.observe() must be called before prepare_request().")
            current = self._frames[-1]
            max_history_images = max(int(self.config.max_images_per_request) - 1, 0)
            if memory_context is not None:
                max_history_images = min(max_history_images, 1)
            history = self._select_history_locked(max_history_images=max_history_images)
            history_frame_ids = tuple(frame.frame_id for frame in history)

        width = int(current.image_bgr.shape[1])
        height = int(current.image_bgr.shape[0])
        body = self._build_request_body(
            instruction=self._instruction,
            current=current,
            history=history,
            memory_context=memory_context,
            events=dict(events or {}),
        )
        return System2Request(
            frame_id=int(current.frame_id),
            width=width,
            height=height,
            history_frame_ids=history_frame_ids,
            body=body,
        )

    def execute_request(self, request: System2Request) -> System2SessionResult:
        if self.config.mode == "mock":
            return self._execute_mock(request)

        try:
            first_round = self._execute_llm_round(
                request.body,
                width=int(request.width),
                height=int(request.height),
                history_frame_ids=tuple(request.history_frame_ids),
            )
            if not first_round.ok or first_round.decision is None or first_round.decision.mode != "look_down":
                return first_round

            follow_up_body = self._build_follow_up_body(request=request, first_round=first_round)
            second_round = self._execute_llm_round(
                follow_up_body,
                width=int(request.width),
                height=int(request.height),
                history_frame_ids=tuple(request.history_frame_ids),
            )
            total_latency_ms = float(first_round.latency_ms + second_round.latency_ms)
            if not second_round.ok or second_round.decision is None:
                return System2SessionResult(
                    ok=False,
                    latency_ms=total_latency_ms,
                    error=str(second_round.error),
                    source=str(second_round.source),
                )

            final_decision = second_round.decision
            if final_decision.mode in {"look_down", "wait"}:
                return System2SessionResult(
                    ok=True,
                    decision=System2Decision(
                        mode="wait",
                        reason=str(final_decision.reason or "system2_follow_up_unresolved"),
                        raw_text=str(final_decision.raw_text),
                        history_frame_ids=tuple(request.history_frame_ids),
                        needs_requery=True,
                    ),
                    latency_ms=total_latency_ms,
                    source="llm",
                )

            return System2SessionResult(
                ok=True,
                decision=final_decision,
                latency_ms=total_latency_ms,
                source="llm",
            )
        except Exception as exc:  # noqa: BLE001
            return System2SessionResult(
                ok=False,
                latency_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
                source="llm",
            )

    def record_result(self, result: System2SessionResult) -> None:
        if not result.ok or result.decision is None:
            return
        with self._lock:
            self._last_output = str(result.decision.raw_text)
            self._last_reason = str(result.decision.reason)
            self._last_history_frame_ids = tuple(result.decision.history_frame_ids)
            self._last_decision_mode = result.decision.mode
            self._last_needs_requery = bool(result.decision.needs_requery)

    def debug_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "instruction": self._instruction,
                "config": {
                    "num_history": int(self.config.num_history),
                    "max_images_per_request": int(self.config.max_images_per_request),
                },
                "observed_frames": [frame.frame_id for frame in self._frames],
                "last_output": self._last_output,
                "last_reason": self._last_reason,
                "last_history_frame_ids": list(self._last_history_frame_ids),
                "last_decision_mode": self._last_decision_mode,
                "last_needs_requery": self._last_needs_requery,
            }

    def _select_history_locked(self, *, max_history_images: int | None = None) -> list[_ObservedFrame]:
        if len(self._frames) <= 1:
            return []
        if max_history_images is not None and int(max_history_images) <= 0:
            return []
        upper = len(self._frames) - 2
        sample_count = min(int(self.config.num_history), upper + 1)
        if max_history_images is not None:
            sample_count = min(sample_count, int(max_history_images))
        if sample_count <= 0:
            return []
        if sample_count == 1:
            indices = [upper]
        else:
            indices = np.unique(np.linspace(0, upper, sample_count, dtype=np.int32)).tolist()
        return [self._frames[index] for index in indices]

    def _lookup_frame_locked(self, frame_id: int) -> _ObservedFrame | None:
        for frame in reversed(self._frames):
            if int(frame.frame_id) == int(frame_id):
                return frame
        return self._frames[-1] if self._frames else None

    def _build_request_body(
        self,
        *,
        instruction: str,
        current: _ObservedFrame,
        history: list[_ObservedFrame],
        memory_context: MemoryContextBundle | None,
        events: dict[str, Any],
    ) -> dict[str, Any]:
        width = int(current.image_bgr.shape[1])
        height = int(current.image_bgr.shape[0])
        prompt_lines = [
            "You are an autonomous navigation assistant.",
            f"Instruction: {instruction}",
            f"Events: {events}",
            f"Current image size: width={width}, height={height}.",
            "The current image is already the waypoint-selection view for navigation.",
        ]
        scratchpad_lines = self._scratchpad_lines(memory_context)
        if scratchpad_lines:
            prompt_lines.append("Scratchpad:")
            prompt_lines.extend(f"- {line}" for line in scratchpad_lines)
        memory_lines = self._memory_lines(memory_context)
        if memory_lines:
            prompt_lines.append("Relevant memory:")
            prompt_lines.extend(f"- {line}" for line in memory_lines)
        else:
            prompt_lines.append("Relevant memory:")
            prompt_lines.append("- None")
        prompt_lines.extend(
            [
                "Respond with exactly one of these formats:",
                "- '<y>, <x>' for a waypoint in the current image",
                "- 'STOP' if the task is complete now",
                "- '←' to request a left turn before selecting a waypoint",
                "- '→' to request a right turn before selecting a waypoint",
                "- '↓' only if you need another frame before deciding",
                "Do not output JSON.",
            ]
        )
        prompt = "\n".join(prompt_lines)

        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if history:
            user_content.append({"type": "text", "text": "Recent history observation:"})
            for frame in history[-1:]:
                user_content.append({"type": "text", "text": f"Recent history (frame {frame.frame_id}):"})
                user_content.append({"type": "image_url", "image_url": {"url": self._encode_image(frame.image_bgr)}})
        if memory_context is not None:
            keyframes = [record for record in memory_context.keyframes if str(record.image_path).strip() != ""][:2]
            if keyframes:
                user_content.append({"type": "text", "text": "Retrieved memory keyframes:"})
                for index, record in enumerate(keyframes, start=1):
                    encoded = self._encode_image_path(record.image_path)
                    if encoded == "":
                        continue
                    label = record.summary or f"Retrieved keyframe {index}"
                    user_content.append({"type": "text", "text": f"Memory keyframe {index}: {label}"})
                    user_content.append({"type": "image_url", "image_url": {"url": encoded}})
            crop_path = str(memory_context.crop_path).strip()
            if crop_path != "":
                encoded_crop = self._encode_image_path(crop_path)
                if encoded_crop != "":
                    user_content.append({"type": "text", "text": "Retrieved memory crop:"})
                    user_content.append({"type": "image_url", "image_url": {"url": encoded_crop}})
        user_content.append({"type": "text", "text": f"Current observation (frame {current.frame_id}):"})
        user_content.append({"type": "image_url", "image_url": {"url": self._encode_image(current.image_bgr)}})

        return {
            "model": self.config.model,
            "temperature": float(self.config.temperature),
            "top_k": int(self.config.top_k),
            "top_p": float(self.config.top_p),
            "min_p": float(self.config.min_p),
            "repeat_penalty": float(self.config.repeat_penalty),
            "max_tokens": 64,
            "messages": [
                {
                    "role": "system",
                    "content": "You must return only a single navigation decision string.",
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        }

    def _build_follow_up_body(self, *, request: System2Request, first_round: System2SessionResult) -> dict[str, Any]:
        if first_round.decision is None:
            raise RuntimeError("System2 follow-up requires a successful first-round decision.")
        with self._lock:
            current = self._lookup_frame_locked(int(request.frame_id))
        if current is None:
            raise RuntimeError(f"Missing observed frame for follow-up: {request.frame_id}")

        base_body = copy.deepcopy(request.body)
        base_messages = list(base_body.get("messages", []))
        if len(base_messages) < 2:
            raise RuntimeError("System2 request body is missing required initial messages.")

        follow_up_content = [
            {
                "type": "text",
                "text": (
                    "You previously answered '↓'. Re-evaluate with this follow-up observation "
                    "and return the final navigation decision now."
                ),
            },
            {"type": "text", "text": f"Follow-up observation (same frame {current.frame_id}):"},
            {"type": "image_url", "image_url": {"url": self._encode_image(current.image_bgr)}},
        ]
        base_messages.append({"role": "assistant", "content": str(first_round.decision.raw_text)})
        base_messages.append({"role": "user", "content": follow_up_content})
        base_body["messages"] = base_messages
        return base_body

    def _execute_llm_round(
        self,
        body: dict[str, Any],
        *,
        width: int,
        height: int,
        history_frame_ids: tuple[int, ...],
    ) -> System2SessionResult:
        start = time.perf_counter()
        try:
            resp = requests.post(
                self.config.endpoint,
                json=body,
                timeout=float(self.config.timeout_sec),
            )
            resp.raise_for_status()
            payload = resp.json()
            raw_text = extract_chat_content(payload)
            decision = parse_system2_output(
                raw_text,
                width=int(width),
                height=int(height),
                history_frame_ids=tuple(history_frame_ids),
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            return System2SessionResult(
                ok=True,
                decision=decision,
                latency_ms=float(latency_ms),
                source="llm",
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            return System2SessionResult(
                ok=False,
                latency_ms=float(latency_ms),
                error=f"{type(exc).__name__}: {exc}",
                source="llm",
            )

    def _execute_mock(self, request: System2Request) -> System2SessionResult:
        start = time.perf_counter()
        width = max(int(request.width), 1)
        height = max(int(request.height), 1)
        decision = System2Decision(
            mode="pixel_goal",
            pixel_goal=(width // 2, int(np.clip(int(0.7 * height), 0, height - 1))),
            reason="mock_forward",
            raw_text=f"{int(0.7 * height)}, {width // 2}",
            history_frame_ids=tuple(request.history_frame_ids),
            needs_requery=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return System2SessionResult(ok=True, decision=decision, latency_ms=float(latency_ms), source="mock")

    @staticmethod
    def _encode_image(image_bgr: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".jpg", np.asarray(image_bgr, dtype=np.uint8))
        if not ok:
            raise ValueError("failed to encode System2 image as JPEG")
        payload = base64.b64encode(io.BytesIO(encoded).getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    @classmethod
    def _encode_image_path(cls, image_path: str) -> str:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return ""
        return cls._encode_image(image)

    @staticmethod
    def _scratchpad_lines(memory_context: MemoryContextBundle | None) -> list[str]:
        if memory_context is None or memory_context.scratchpad is None:
            return []
        scratchpad = memory_context.scratchpad
        lines: list[str] = []
        if scratchpad.goal_summary.strip() != "":
            lines.append(f"Goal: {scratchpad.goal_summary.strip()}")
        if scratchpad.checked_locations:
            lines.append("Checked: " + ", ".join(scratchpad.checked_locations[-3:]))
        if scratchpad.recent_hint.strip() != "":
            lines.append(f"Hint: {scratchpad.recent_hint.strip()}")
        if scratchpad.next_priority.strip() != "":
            lines.append(f"Next: {scratchpad.next_priority.strip()}")
        return lines[:4]

    @staticmethod
    def _memory_lines(memory_context: MemoryContextBundle | None) -> list[str]:
        if memory_context is None:
            return []
        return [str(line.text).strip() for line in memory_context.text_lines if str(line.text).strip() != ""][:5]


def _first_token_index(text: str, candidates: tuple[str, ...]) -> int:
    matches = [text.find(candidate) for candidate in candidates if text.find(candidate) >= 0]
    if not matches:
        return -1
    return min(matches)
