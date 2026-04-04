from __future__ import annotations

import base64
import copy
import io
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inference.vlm.llama_cpp_sidecar import (
    DEFAULT_CACHE_TYPE_K,
    DEFAULT_CACHE_TYPE_V,
    DEFAULT_CTX_SIZE,
    DEFAULT_FLASH_ATTN,
    DEFAULT_GPU_LAYERS,
    DEFAULT_HEALTH_TIMEOUT_S,
    DEFAULT_HOST,
    DEFAULT_LLAMA_URL,
    DEFAULT_MAIN_GPU,
    DEFAULT_MAX_TOKENS,
    LlamaCppSidecar,
)


DEFAULT_PORT = 15801
DEFAULT_RESIZE_W = 384
DEFAULT_RESIZE_H = 384
DEFAULT_NUM_HISTORY = 4
DEFAULT_PLAN_STEP_GAP = 4
DEFAULT_NAVIGATION_ROUTE = "/navigation"
LEGACY_EVAL_DUAL_ROUTE = "/eval_dual"
DEFAULT_CHAT_SESSION_SYSTEM_PROMPT = (
    "You are the dedicated chat assistant attached to the InternVLA navigation server. "
    "Answer conversational requests clearly and concisely. "
    "Do not emit navigation waypoint coordinates or discrete navigation actions unless the user explicitly asks for them."
)
LLAMA_LOOK_DOWN_IMAGE_TOKEN_BUDGET = 196
LLAMA_LOOK_DOWN_TEXT_TOKEN_RESERVE = 320
DEFAULT_PROMPT_MODEL_PATH_CANDIDATES = (
    Path(r"C:\Users\mango\project\pp\deploy\InternVLA-N1-System2-4bit"),
)
PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed the task."
)
CONJUNCTIONS = (
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
)
ACTION_TO_ID = {
    "STOP": [0],
    "↑": [1],
    "←": [2],
    "→": [3],
    "↓": [5],
}
ACTION_ID_TO_MODE = {
    0: "stop",
    1: "forward",
    2: "yaw_left",
    3: "yaw_right",
    5: "wait",
}
TURN_LEFT_TOKENS = ("turn left", "yaw left", "left", "←")
TURN_RIGHT_TOKENS = ("turn right", "yaw right", "right", "→")
FORWARD_TOKENS = ("move forward", "go forward", "forward", "ahead", "↑")
LOOK_DOWN_TOKENS = ("look down", "tilt down", "down", "↓")


def _bool_arg(value: Any) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _extract_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"llama.cpp response does not include choices: {response}")
    first_choice = choices[0]
    if isinstance(first_choice, dict) and "message" in first_choice:
        content = first_choice.get("message", {}).get("content")
    elif isinstance(first_choice, dict):
        content = first_choice.get("text", "")
    else:
        content = first_choice
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        return "".join(texts).strip()
    return str(content or "").strip()


def _normalize_chat_session_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        normalized = content.strip()
        if not normalized:
            raise ValueError("message must not be empty.")
        return normalized
    if isinstance(content, dict):
        return [copy.deepcopy(content)]
    if isinstance(content, list):
        if not content:
            raise ValueError("message must not be empty.")
        return copy.deepcopy(content)
    raise ValueError("message must be a string, an object, or a list of content blocks.")


def _first_token_index(text: str, candidates: tuple[str, ...]) -> int:
    matches = [text.find(candidate) for candidate in candidates if text.find(candidate) >= 0]
    return -1 if not matches else min(matches)


def _parse_navigation_output(output: str) -> tuple[str, list[int] | None, list[int] | None, bool]:
    text = str(output).strip()
    lowered = text.lower()
    numbers = [int(token) for token in re.findall(r"-?\d+", text)]
    if len(numbers) >= 2:
        return "pixel_goal", None, [int(numbers[1]), int(numbers[0])], False
    if re.search(r"\bstop\b", lowered):
        return "stop", [0], None, False
    left_index = _first_token_index(lowered, TURN_LEFT_TOKENS)
    right_index = _first_token_index(lowered, TURN_RIGHT_TOKENS)
    if left_index >= 0 and (right_index < 0 or left_index <= right_index):
        return "yaw_left", [2], None, False
    if right_index >= 0:
        return "yaw_right", [3], None, False
    if _first_token_index(lowered, FORWARD_TOKENS) >= 0:
        return "forward", [1], None, False
    if _first_token_index(lowered, LOOK_DOWN_TOKENS) >= 0:
        return "look_down", [5], None, True
    return "wait", [5], None, True


def _resolve_prompt_model_path(raw_value: Any) -> Path | None:
    if raw_value is not None and str(raw_value).strip():
        resolved = Path(str(raw_value)).expanduser().resolve()
        if resolved.exists():
            return resolved
    for candidate in DEFAULT_PROMPT_MODEL_PATH_CANDIDATES:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    return None


def _pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@dataclass(slots=True)
class ChatSessionState:
    session_id: str
    messages: list[dict[str, Any]]
    created_at_s: float
    updated_at_s: float
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(slots=True)
class NavigationSessionState:
    session_id: str
    instruction: str = ""
    language: str = "auto"
    rgb_list: list[Image.Image] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)
    last_s2_idx: int = -100
    llm_output: str = ""
    output_action: list[int] | None = None
    output_pixel: list[int] | None = None
    last_decision_mode: str = "wait"
    last_raw_output_text: str = ""
    last_history_frame_ids: list[int] = field(default_factory=list)
    last_needs_requery: bool = False
    last_latency_ms: float = 0.0
    pixel_goal_count: int = 0
    action_only_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset(self, instruction: str, *, language: str) -> None:
        self.instruction = str(instruction).strip()
        self.language = str(language).strip() or "auto"
        self.rgb_list.clear()
        self.frame_ids.clear()
        self.last_s2_idx = -100
        self.llm_output = ""
        self.output_action = None
        self.output_pixel = None
        self.last_decision_mode = "wait"
        self.last_raw_output_text = ""
        self.last_history_frame_ids = []
        self.last_needs_requery = False
        self.last_latency_ms = 0.0
        self.pixel_goal_count = 0
        self.action_only_count = 0


class System2Service:
    def __init__(self, args, *, sidecar: LlamaCppSidecar | None = None) -> None:
        self.host = str(getattr(args, "host", DEFAULT_HOST))
        self.port = int(getattr(args, "port", DEFAULT_PORT))
        self.resize_w = int(getattr(args, "resize_w", DEFAULT_RESIZE_W))
        self.resize_h = int(getattr(args, "resize_h", DEFAULT_RESIZE_H))
        self.num_history = int(getattr(args, "num_history", DEFAULT_NUM_HISTORY))
        self.plan_step_gap = int(getattr(args, "plan_step_gap", DEFAULT_PLAN_STEP_GAP))
        self.max_new_tokens = int(getattr(args, "max_new_tokens", DEFAULT_MAX_TOKENS))
        self.chat_session_system_prompt = (
            str(getattr(args, "chat_session_system_prompt", DEFAULT_CHAT_SESSION_SYSTEM_PROMPT)).strip()
            or DEFAULT_CHAT_SESSION_SYSTEM_PROMPT
        )
        self._navigation_sessions_lock = threading.Lock()
        self._navigation_sessions: dict[str, NavigationSessionState] = {}
        self._chat_sessions_lock = threading.Lock()
        self._chat_sessions: dict[str, ChatSessionState] = {}
        prompt_model_path = _resolve_prompt_model_path(getattr(args, "prompt_model_path", None))
        self._sidecar = sidecar or LlamaCppSidecar(
            llama_cpp_root=Path(args.llama_cpp_root),
            llama_server_path=Path(args.llama_server_path),
            llama_model_path=Path(args.llama_model_path),
            llama_mmproj_path=Path(args.llama_mmproj_path),
            llama_url=str(getattr(args, "llama_url", DEFAULT_LLAMA_URL)),
            prompt_model_path=prompt_model_path,
            ctx_size=int(getattr(args, "llama_ctx_size", DEFAULT_CTX_SIZE)),
            threads=getattr(args, "llama_threads", None),
            gpu_layers=str(getattr(args, "llama_gpu_layers", DEFAULT_GPU_LAYERS)),
            main_gpu=int(getattr(args, "llama_main_gpu", DEFAULT_MAIN_GPU)),
            max_tokens=int(getattr(args, "max_new_tokens", DEFAULT_MAX_TOKENS)),
            flash_attn=str(getattr(args, "llama_flash_attn", DEFAULT_FLASH_ATTN)),
            cache_type_k=getattr(args, "llama_cache_type_k", DEFAULT_CACHE_TYPE_K),
            cache_type_v=getattr(args, "llama_cache_type_v", DEFAULT_CACHE_TYPE_V),
            cache_prompt=_bool_arg(getattr(args, "llama_cache_prompt", "0")),
            chat_lora_path=getattr(args, "llama_chat_lora_path", None),
            chat_lora_scale=float(getattr(args, "llama_chat_lora_scale", 1.0)),
            stdout_log=Path(getattr(args, "llama_stdout_log")),
            stderr_log=Path(getattr(args, "llama_stderr_log")),
            health_timeout_s=float(getattr(args, "llama_health_timeout_s", DEFAULT_HEALTH_TIMEOUT_S)),
        )
        max_look_down_images = max(
            2,
            (int(getattr(args, "llama_ctx_size", DEFAULT_CTX_SIZE)) - LLAMA_LOOK_DOWN_TEXT_TOKEN_RESERVE)
            // LLAMA_LOOK_DOWN_IMAGE_TOKEN_BUDGET,
        )
        self.max_look_down_images = max(2, min(self.num_history + 2, max_look_down_images))

    def close(self) -> None:
        with self._navigation_sessions_lock:
            self._navigation_sessions.clear()
        with self._chat_sessions_lock:
            self._chat_sessions.clear()
        self._sidecar.close()

    def ready(self) -> bool:
        return self._sidecar.healthy()

    def health_payload(self) -> dict[str, Any]:
        latest_output = None
        with self._navigation_sessions_lock:
            if self._navigation_sessions:
                latest_session = max(self._navigation_sessions.values(), key=lambda item: item.last_latency_ms)
                latest_output = {
                    "instruction": latest_session.instruction,
                    "decisionMode": latest_session.last_decision_mode,
                    "rawText": latest_session.last_raw_output_text,
                    "historyFrameIds": list(latest_session.last_history_frame_ids),
                    "needsRequery": bool(latest_session.last_needs_requery),
                    "latencyMs": float(latest_session.last_latency_ms),
                }
        payload = {
            "status": "ok",
            "ready": self.ready(),
            "service": "system2_wrapper",
            "backend": "llama_cpp",
            "host": self.host,
            "port": self.port,
            "navigation_route": DEFAULT_NAVIGATION_ROUTE,
            "resize_w": self.resize_w,
            "resize_h": self.resize_h,
            "num_history": self.num_history,
            "plan_step_gap": self.plan_step_gap,
            "max_new_tokens": self.max_new_tokens,
            "navigation_session_count": len(self._navigation_sessions),
            "chat_session_count": len(self._chat_sessions),
            "chat_session_system_prompt": self.chat_session_system_prompt,
            "system2_output": latest_output,
        }
        payload.update(self._sidecar.health_payload())
        return payload

    def _get_navigation_session(self, session_id: str) -> NavigationSessionState:
        with self._navigation_sessions_lock:
            session = self._navigation_sessions.get(session_id)
            if session is None:
                session = NavigationSessionState(session_id=session_id)
                self._navigation_sessions[session_id] = session
            return session

    def _select_history_indices(self, session: NavigationSessionState) -> list[int]:
        if len(session.rgb_list) <= 1:
            return []
        upper = len(session.rgb_list) - 2
        sample_count = min(int(self.num_history), upper + 1)
        if sample_count <= 0:
            return []
        if sample_count == 1:
            return [upper]
        return np.unique(np.linspace(0, upper, sample_count, dtype=np.int32)).tolist()

    def _build_primary_messages(
        self,
        *,
        session: NavigationSessionState,
        history_indices: list[int],
        current_image: Image.Image,
    ) -> list[dict[str, Any]]:
        current_frame_id = session.frame_ids[-1]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": PROMPT_TEMPLATE.replace("<instruction>", session.instruction),
            }
        ]
        if history_indices:
            content.append({"type": "text", "text": "These are your historical observations:"})
            for index in history_indices:
                content.append({"type": "text", "text": f"Historical observation (frame {session.frame_ids[index]}):"})
                content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(session.rgb_list[index])}})
        content.append(
            {
                "type": "text",
                "text": f"{CONJUNCTIONS[0]} Current observation (frame {current_frame_id}):",
            }
        )
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(current_image)}})
        return [
            {"role": "system", "content": "You must return only a single navigation decision string."},
            {"role": "user", "content": content},
        ]

    def _build_look_down_messages(
        self,
        *,
        session: NavigationSessionState,
        current_image: Image.Image,
    ) -> list[dict[str, Any]]:
        reference_cap = max(1, self.max_look_down_images - 1)
        reference_indices = list(range(max(0, len(session.rgb_list) - reference_cap), len(session.rgb_list)))
        current_key = reference_indices[-1]
        history_indices = reference_indices[:-1]
        messages = self._build_primary_messages(
            session=session,
            history_indices=history_indices,
            current_image=session.rgb_list[current_key],
        )
        messages.append({"role": "assistant", "content": [{"type": "text", "text": session.llm_output}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You previously answered 'look down'. Re-evaluate with this follow-up observation "
                            "and return the final navigation decision now."
                        ),
                    },
                    {"type": "text", "text": f"Follow-up observation (frame {session.frame_ids[-1]}):"},
                    {"type": "image_url", "image_url": {"url": _pil_to_data_url(current_image)}},
                ],
            }
        )
        session.last_history_frame_ids = [session.frame_ids[index] for index in history_indices]
        return messages

    def _record_decision(
        self,
        *,
        session: NavigationSessionState,
        llm_output: str,
        output_action: list[int] | None,
        output_pixel: list[int] | None,
        history_frame_ids: list[int],
        latency_ms: float,
        needs_requery: bool,
    ) -> None:
        session.last_raw_output_text = str(llm_output)
        session.last_history_frame_ids = list(history_frame_ids)
        session.last_latency_ms = float(latency_ms)
        session.last_needs_requery = bool(needs_requery)
        if output_pixel is not None:
            session.last_decision_mode = "pixel_goal"
            session.pixel_goal_count += 1
            return
        if output_action:
            session.last_decision_mode = ACTION_ID_TO_MODE.get(int(output_action[0]), "wait")
            session.action_only_count += 1
            return
        session.last_decision_mode = "wait"
        session.action_only_count += 1

    def _run_completion(self, messages: list[dict[str, Any]], *, lora: list[dict[str, Any]] | None = None) -> tuple[str, float]:
        start = time.monotonic()
        response = self._sidecar.chat_completion(messages=messages, max_tokens=self.max_new_tokens, lora=lora)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return _extract_completion_text(response), float(elapsed_ms)

    def _step_s2(self, *, session: NavigationSessionState, current_image: Image.Image) -> tuple[list[int] | None, list[int] | None]:
        history_indices = self._select_history_indices(session)
        session.last_history_frame_ids = [session.frame_ids[index] for index in history_indices]
        messages = self._build_primary_messages(
            session=session,
            history_indices=history_indices,
            current_image=current_image,
        )
        llm_output, latency_ms = self._run_completion(messages)
        session.llm_output = llm_output
        _decision_mode, output_action, output_pixel, needs_requery = _parse_navigation_output(llm_output)
        self._record_decision(
            session=session,
            llm_output=llm_output,
            output_action=output_action,
            output_pixel=output_pixel,
            history_frame_ids=session.last_history_frame_ids,
            latency_ms=latency_ms,
            needs_requery=needs_requery,
        )
        if output_action == [5]:
            follow_up_messages = self._build_look_down_messages(session=session, current_image=current_image)
            llm_output, latency_ms = self._run_completion(follow_up_messages)
            session.llm_output = llm_output
            _decision_mode, output_action, output_pixel, needs_requery = _parse_navigation_output(llm_output)
            self._record_decision(
                session=session,
                llm_output=llm_output,
                output_action=output_action,
                output_pixel=output_pixel,
                history_frame_ids=session.last_history_frame_ids,
                latency_ms=latency_ms,
                needs_requery=needs_requery,
            )
            if output_action == [5] and output_pixel is None:
                session.output_action = [5]
                session.output_pixel = None
                return [5], None
        session.output_action = None if output_action is None else list(output_action)
        session.output_pixel = None if output_pixel is None else list(output_pixel)
        return session.output_action, session.output_pixel

    @staticmethod
    def _decode_rgb(image_file) -> Image.Image:
        stream = getattr(image_file, "stream", None) or getattr(image_file, "file", None) or image_file
        return Image.open(stream).convert("RGB")

    @staticmethod
    def _decode_depth(depth_file) -> np.ndarray:
        stream = getattr(depth_file, "stream", None) or getattr(depth_file, "file", None) or depth_file
        depth = Image.open(stream).convert("I")
        return np.asarray(depth, dtype=np.float32) / 10000.0

    def eval_dual(self, image_file, depth_file, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip() or f"nav-{uuid.uuid4().hex}"
        instruction = " ".join(str(payload.get("instruction", "")).strip().split())
        if instruction == "":
            raise ValueError("instruction is required.")
        language = str(payload.get("language", "auto")).strip() or "auto"
        idx = int(payload.get("idx", 0))
        reset = bool(payload.get("reset", False))

        rgb = self._decode_rgb(image_file)
        _ = self._decode_depth(depth_file)

        session = self._get_navigation_session(session_id)
        with session.lock:
            if reset or session.instruction != instruction:
                session.reset(instruction, language=language)
            session.rgb_list.append(rgb)
            session.frame_ids.append(idx)

            if session.output_action is not None or session.output_pixel is not None:
                if idx - int(session.last_s2_idx) < int(self.plan_step_gap):
                    return self._normalize_output(session.output_action, session.output_pixel)

            output_action, output_pixel = self._step_s2(session=session, current_image=rgb)
            session.last_s2_idx = int(idx)
            return self._normalize_output(output_action, output_pixel)

    @staticmethod
    def _normalize_output(output_action: list[int] | None, output_pixel: list[int] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if output_action is not None:
            payload["discrete_action"] = [int(value) for value in list(output_action)]
            return payload
        if output_pixel is not None:
            pixel = np.asarray(output_pixel, dtype=np.int32).reshape(2)
            payload["pixel_goal"] = [int(pixel[0]), int(pixel[1])]
            return payload
        return {"discrete_action": [5]}

    def open_chat_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip() or f"chat-{uuid.uuid4().hex}"
        session_system_prompt = str(payload.get("system_prompt", "")).strip()
        lora_request = self._sidecar.chat_lora_request() if self._sidecar.chat_lora_path is not None else []
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.chat_session_system_prompt}]
        if session_system_prompt:
            messages.append({"role": "system", "content": session_system_prompt})
        now = time.monotonic()
        with self._chat_sessions_lock:
            if session_id in self._chat_sessions:
                raise ValueError(f"Chat session already exists: {session_id}")
            self._chat_sessions[session_id] = ChatSessionState(
                session_id=session_id,
                messages=messages,
                created_at_s=now,
                updated_at_s=now,
            )
        return {
            "status": "ok",
            "session_id": session_id,
            "message_count": len(messages),
            "chat_lora_active": bool(lora_request),
            "chat_lora_id": None if not lora_request else lora_request[0]["id"],
            "chat_session_system_prompt": self.chat_session_system_prompt,
            "session_system_prompt": session_system_prompt or None,
        }

    def chat_session_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")
        max_tokens = int(payload.get("max_tokens", self.max_new_tokens))
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        message_content = _normalize_chat_session_content(payload.get("message"))
        with self._chat_sessions_lock:
            session = self._chat_sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown chat session: {session_id}")
        lora_request = self._sidecar.chat_lora_request() if self._sidecar.chat_lora_path is not None else []
        with session.lock:
            user_message = {"role": "user", "content": message_content}
            request_messages = copy.deepcopy(session.messages)
            request_messages.append(copy.deepcopy(user_message))
            response = self._sidecar.chat_completion(
                messages=request_messages,
                max_tokens=max_tokens,
                lora=lora_request or None,
            )
            assistant_text = _extract_completion_text(response)
            session.messages.append(user_message)
            session.messages.append({"role": "assistant", "content": assistant_text})
            session.updated_at_s = time.monotonic()
        return {
            "status": "ok",
            "session_id": session_id,
            "message": {"role": "assistant", "content": assistant_text},
            "message_count": len(session.messages),
            "chat_lora_active": bool(lora_request),
            "chat_lora_id": None if not lora_request else lora_request[0]["id"],
        }

    def close_chat_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")
        with self._chat_sessions_lock:
            session = self._chat_sessions.pop(session_id, None)
        if session is None:
            raise ValueError(f"Unknown chat session: {session_id}")
        return {
            "status": "ok",
            "session_id": session_id,
            "closed": True,
            "message_count": len(session.messages),
        }
