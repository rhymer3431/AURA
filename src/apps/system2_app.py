from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from server.system2_service import (
    DEFAULT_CACHE_TYPE_K,
    DEFAULT_CACHE_TYPE_V,
    DEFAULT_CHAT_SESSION_SYSTEM_PROMPT,
    DEFAULT_CTX_SIZE,
    DEFAULT_FLASH_ATTN,
    DEFAULT_GPU_LAYERS,
    DEFAULT_HEALTH_TIMEOUT_S,
    DEFAULT_HOST,
    DEFAULT_LLAMA_URL,
    DEFAULT_MAIN_GPU,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NAVIGATION_ROUTE,
    DEFAULT_NUM_HISTORY,
    DEFAULT_PLAN_STEP_GAP,
    DEFAULT_PORT,
    DEFAULT_RESIZE_H,
    DEFAULT_RESIZE_W,
    LEGACY_EVAL_DUAL_ROUTE,
    System2Service,
)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(str(raw).strip())


def _env_optional_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrapper-backed System2 service for llama.cpp navigation + chat.")
    parser.add_argument("--host", type=str, default=_env_str("INTERNVLA_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=_env_int("INTERNVLA_PORT", DEFAULT_PORT))
    parser.add_argument("--resize-w", type=int, default=_env_int("INTERNVLA_RESIZE_W", DEFAULT_RESIZE_W))
    parser.add_argument("--resize-h", type=int, default=_env_int("INTERNVLA_RESIZE_H", DEFAULT_RESIZE_H))
    parser.add_argument("--num-history", type=int, default=_env_int("INTERNVLA_NUM_HISTORY", DEFAULT_NUM_HISTORY))
    parser.add_argument("--plan-step-gap", type=int, default=_env_int("INTERNVLA_PLAN_STEP_GAP", DEFAULT_PLAN_STEP_GAP))
    parser.add_argument("--max-new-tokens", type=int, default=_env_int("INTERNVLA_MAX_NEW_TOKENS", DEFAULT_MAX_TOKENS))
    parser.add_argument(
        "--prompt-model-path",
        type=Path,
        default=Path(_env_optional_str("INTERNVLA_PROMPT_MODEL_PATH")).expanduser()
        if _env_optional_str("INTERNVLA_PROMPT_MODEL_PATH")
        else None,
    )
    parser.add_argument(
        "--llama-cpp-root",
        type=Path,
        default=Path(_env_str("INTERNVLA_LLAMA_CPP_ROOT", str(Path(__file__).resolve().parents[2] / "llama.cpp"))),
    )
    parser.add_argument(
        "--llama-server-path",
        type=Path,
        default=Path(
            _env_str("INTERNVLA_LLAMA_SERVER_PATH", str(Path(__file__).resolve().parents[2] / "llama.cpp" / "llama-server.exe"))
        ),
    )
    parser.add_argument(
        "--llama-model-path",
        type=Path,
        default=Path(
            _env_str(
                "INTERNVLA_LLAMA_MODEL_PATH",
                str(Path(__file__).resolve().parents[2] / "artifacts" / "models" / "InternVLA-N1-System2.Q4_K_M.gguf"),
            )
        ),
    )
    parser.add_argument(
        "--llama-mmproj-path",
        type=Path,
        default=Path(
            _env_str(
                "INTERNVLA_LLAMA_MMPROJ_PATH",
                str(Path(__file__).resolve().parents[2] / "artifacts" / "models" / "InternVLA-N1-System2.mmproj-Q8_0.gguf"),
            )
        ),
    )
    parser.add_argument("--llama-url", type=str, default=_env_str("INTERNVLA_LLAMA_URL", DEFAULT_LLAMA_URL))
    parser.add_argument("--llama-ctx-size", type=int, default=_env_int("INTERNVLA_LLAMA_CTX_SIZE", DEFAULT_CTX_SIZE))
    parser.add_argument("--llama-threads", default=_env_optional_str("INTERNVLA_LLAMA_THREADS"))
    parser.add_argument("--llama-gpu-layers", default=_env_str("INTERNVLA_LLAMA_GPU_LAYERS", DEFAULT_GPU_LAYERS))
    parser.add_argument("--llama-main-gpu", type=int, default=_env_int("INTERNVLA_LLAMA_MAIN_GPU", DEFAULT_MAIN_GPU))
    parser.add_argument("--llama-flash-attn", default=_env_str("INTERNVLA_LLAMA_FLASH_ATTN", DEFAULT_FLASH_ATTN))
    parser.add_argument("--llama-cache-type-k", default=_env_str("INTERNVLA_LLAMA_CACHE_TYPE_K", DEFAULT_CACHE_TYPE_K))
    parser.add_argument("--llama-cache-type-v", default=_env_str("INTERNVLA_LLAMA_CACHE_TYPE_V", DEFAULT_CACHE_TYPE_V))
    parser.add_argument("--llama-cache-prompt", default=_env_str("INTERNVLA_PROMPT_CACHE", "0"))
    parser.add_argument(
        "--llama-chat-lora-path",
        type=Path,
        default=Path(_env_optional_str("INTERNVLA_LLAMA_CHAT_LORA_PATH")).expanduser()
        if _env_optional_str("INTERNVLA_LLAMA_CHAT_LORA_PATH")
        else None,
    )
    parser.add_argument("--llama-chat-lora-scale", type=float, default=float(_env_str("INTERNVLA_LLAMA_CHAT_LORA_SCALE", "1.0")))
    parser.add_argument("--chat-session-system-prompt", default=_env_str("INTERNVLA_CHAT_SESSION_SYSTEM_PROMPT", DEFAULT_CHAT_SESSION_SYSTEM_PROMPT))
    parser.add_argument("--llama-health-timeout-s", type=float, default=float(_env_str("INTERNVLA_LLAMA_HEALTH_TIMEOUT_S", str(DEFAULT_HEALTH_TIMEOUT_S))))
    parser.add_argument(
        "--llama-stdout-log",
        type=Path,
        default=Path(
            _env_str(
                "INTERNVLA_LLAMA_STDOUT_LOG",
                str(Path(__file__).resolve().parents[2] / "tmp" / "process_logs" / "system" / "internvla_llama.stdout.log"),
            )
        ),
    )
    parser.add_argument(
        "--llama-stderr-log",
        type=Path,
        default=Path(
            _env_str(
                "INTERNVLA_LLAMA_STDERR_LOG",
                str(Path(__file__).resolve().parents[2] / "tmp" / "process_logs" / "system" / "internvla_llama.stderr.log"),
            )
        ),
    )
    return parser.parse_known_args(argv)[0]


def _json_request_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        raise ValueError("Request body must decode to an object.")
    return payload


def create_app(args: argparse.Namespace | None = None, *, service: System2Service | None = None) -> Flask:
    parsed_args = parse_args([]) if args is None else args
    app = Flask(__name__)
    runtime = service or System2Service(parsed_args)
    app.config["SYSTEM2_SERVICE"] = runtime

    @app.route("/healthz", methods=["GET"])
    def healthz() -> Any:
        payload = runtime.health_payload()
        if bool(payload.get("ready", False)):
            return jsonify(payload), 200
        error_payload = dict(payload)
        error_payload["status"] = "error"
        error_payload["ready"] = False
        return jsonify(error_payload), 503

    @app.route(DEFAULT_NAVIGATION_ROUTE, methods=["POST"])
    @app.route(LEGACY_EVAL_DUAL_ROUTE, methods=["POST"])
    def navigation() -> Any:
        image_file = request.files.get("image")
        depth_file = request.files.get("depth")
        raw_json = request.form.get("json", "")
        if image_file is None:
            return jsonify({"status": "error", "message": "Missing multipart image file."}), 400
        if depth_file is None:
            return jsonify({"status": "error", "message": "Missing multipart depth file."}), 400
        try:
            payload = json.loads(raw_json) if raw_json else {}
        except json.JSONDecodeError as exc:
            return jsonify({"status": "error", "message": f"Invalid json form field: {exc}"}), 400
        if not isinstance(payload, dict):
            return jsonify({"status": "error", "message": "json form field must decode to an object."}), 400
        try:
            return jsonify(runtime.eval_dual(image_file, depth_file, payload))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    @app.route("/chat/session/open", methods=["POST"])
    def chat_session_open() -> Any:
        try:
            return jsonify(runtime.open_chat_session(_json_request_payload()))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    @app.route("/chat/session/message", methods=["POST"])
    def chat_session_message() -> Any:
        try:
            return jsonify(runtime.chat_session_message(_json_request_payload()))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    @app.route("/chat/session/close", methods=["POST"])
    def chat_session_close() -> Any:
        try:
            return jsonify(runtime.close_chat_session(_json_request_payload()))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    return app


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(args)
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
