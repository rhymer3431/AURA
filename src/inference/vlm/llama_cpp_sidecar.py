from __future__ import annotations

import copy
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

DEFAULT_HOST = "127.0.0.1"
DEFAULT_LLAMA_URL = "http://127.0.0.1:8081"
DEFAULT_CTX_SIZE = 8192
DEFAULT_MAX_TOKENS = 16
DEFAULT_FLASH_ATTN = "on"
DEFAULT_GPU_LAYERS = "all"
DEFAULT_MAIN_GPU = 0
DEFAULT_CACHE_TYPE_K = "q8_0"
DEFAULT_CACHE_TYPE_V = "q8_0"
DEFAULT_HEALTH_TIMEOUT_S = 300.0
DEFAULT_HEALTH_POLL_TIMEOUT_S = 0.25
DEFAULT_REQUEST_TIMEOUT_S = 30.0
HF_QWEN_IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
HF_QWEN_VIDEO_MARKER = "<|vision_start|><|video_pad|><|vision_end|>"
LLAMA_MTMD_MEDIA_MARKER = "<__media__>"


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    texts.append(maybe_text)
        return "".join(texts).strip()
    if isinstance(content, dict):
        maybe_text = content.get("text")
        if isinstance(maybe_text, str):
            return maybe_text.strip()
    return str(content or "").strip()


def _normalize_cache_type(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.lower() in {"none", "off", "disable", "disabled", "false", "0"}:
        return None
    return normalized


def _normalize_llama_url(value: str) -> str:
    raw = str(value).strip().rstrip("/")
    if not raw:
        raise ValueError("llama_url is required.")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"llama_url must be absolute http(s): {value!r}")
    return raw


def _llama_url_host_port(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    host = parsed.hostname or DEFAULT_HOST
    port = parsed.port or 8081
    return host, int(port)


def _data_url_to_base64_payload(url: str) -> str:
    text = str(url).strip()
    if not text:
        raise ValueError("image_url cannot be empty")
    if not text.startswith("data:"):
        raise ValueError("Only data:image/...;base64 URLs are supported for low-level multimodal prompts")
    _, _, payload = text.partition(",")
    if not payload:
        raise ValueError("image_url data URL does not include a base64 payload")
    return payload


def _build_hf_chat_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user") or "user").strip() or "user"
        content = message.get("content", "")
        if isinstance(content, str):
            serialized.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            serialized.append({"role": role, "content": _extract_text_content(content)})
            continue
        parts: list[dict[str, str]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                parts.append({"type": "image"})
                continue
            text_value = ""
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_value = item.get("text", "")
            elif isinstance(item, str):
                text_value = item
            else:
                text_value = _extract_text_content(item)
            if text_value:
                parts.append({"type": "text", "text": text_value})
        serialized.append({"role": role, "content": parts})
    return serialized


def _serialize_hf_chat_template_prompt(
    processor: Any,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_messages = _build_hf_chat_template_messages(messages)
    prompt_string = str(
        processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    )
    prompt_string = prompt_string.replace(HF_QWEN_IMAGE_MARKER, LLAMA_MTMD_MEDIA_MARKER)
    prompt_string = prompt_string.replace(HF_QWEN_VIDEO_MARKER, LLAMA_MTMD_MEDIA_MARKER)

    multimodal_data: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url", {})
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if isinstance(url, str) and url.strip():
                multimodal_data.append(_data_url_to_base64_payload(url))

    payload: dict[str, Any] = {"prompt_string": prompt_string}
    if multimodal_data:
        payload["multimodal_data"] = multimodal_data
    return payload


def _build_llama_server_command(
    *,
    server_exe: Path,
    host: str,
    port: int,
    model_path: Path,
    mmproj_path: Path,
    ctx_size: int,
    threads: str | None,
    gpu_layers: str,
    main_gpu: int,
    max_tokens: int,
    flash_attn: str,
    cache_type_k: str | None,
    cache_type_v: str | None,
    cache_prompt: bool,
    chat_lora_path: Path | None,
) -> list[str]:
    command = [
        str(server_exe),
        "--host",
        str(host),
        "--port",
        str(port),
        "--model",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--flash-attn",
        str(flash_attn),
        "--ctx-size",
        str(int(ctx_size)),
        "--embeddings",
        "--predict",
        str(int(max_tokens)),
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--no-warmup",
        "--main-gpu",
        str(int(main_gpu)),
    ]
    if threads is not None and str(threads).strip():
        command.extend(["--threads", str(threads).strip()])
    if str(gpu_layers).strip():
        command.extend(["--gpu-layers", str(gpu_layers).strip()])
    if cache_prompt:
        command.append("--cache-prompt")
    else:
        command.append("--no-cache-prompt")
    normalized_k = _normalize_cache_type(cache_type_k)
    if normalized_k is not None:
        command.extend(["--cache-type-k", normalized_k])
    normalized_v = _normalize_cache_type(cache_type_v)
    if normalized_v is not None:
        command.extend(["--cache-type-v", normalized_v])
    if chat_lora_path is not None:
        command.extend(["--lora", str(chat_lora_path), "--lora-init-without-apply"])
    return command


class LlamaCppSidecar:
    def __init__(
        self,
        *,
        llama_cpp_root: Path,
        llama_server_path: Path,
        llama_model_path: Path,
        llama_mmproj_path: Path,
        llama_url: str = DEFAULT_LLAMA_URL,
        prompt_model_path: Path | None = None,
        ctx_size: int = DEFAULT_CTX_SIZE,
        threads: str | None = None,
        gpu_layers: str = DEFAULT_GPU_LAYERS,
        main_gpu: int = DEFAULT_MAIN_GPU,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        flash_attn: str = DEFAULT_FLASH_ATTN,
        cache_type_k: str | None = DEFAULT_CACHE_TYPE_K,
        cache_type_v: str | None = DEFAULT_CACHE_TYPE_V,
        cache_prompt: bool = False,
        chat_lora_path: Path | None = None,
        chat_lora_scale: float = 1.0,
        stdout_log: Path | None = None,
        stderr_log: Path | None = None,
        health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
        auto_start: bool = True,
    ) -> None:
        self.root = Path(llama_cpp_root).expanduser().resolve()
        self.server_exe = Path(llama_server_path).expanduser().resolve()
        self.model_path = Path(llama_model_path).expanduser().resolve()
        self.mmproj_path = Path(llama_mmproj_path).expanduser().resolve()
        self.base_url = _normalize_llama_url(str(llama_url))
        self.host, self.port = _llama_url_host_port(self.base_url)
        self.model_name = self.model_path.name
        self.ctx_size = int(ctx_size)
        self.health_timeout_s = float(health_timeout_s)
        self.embeddings = True
        self.cache_type_k = _normalize_cache_type(cache_type_k)
        self.cache_type_v = _normalize_cache_type(cache_type_v)
        self.cache_prompt = bool(cache_prompt)
        self.chat_lora_path = None if chat_lora_path is None else Path(chat_lora_path).expanduser().resolve()
        self.chat_lora_scale = float(chat_lora_scale)
        self.prompt_model_path = self._resolve_prompt_model_path(prompt_model_path)
        self.stdout_log = Path(stdout_log).resolve() if stdout_log is not None else self.root / "internvla_llama_stdout.log"
        self.stderr_log = Path(stderr_log).resolve() if stderr_log is not None else self.root / "internvla_llama_stderr.log"
        self._hf_prompt_processor = None
        self._hf_prompt_load_attempted = False
        self._hf_prompt_error: str | None = None
        self._hf_low_level_completions_enabled = True
        self._lora_adapters_cache: list[dict[str, Any]] | None = None
        self._chat_lora_id: int | None = None
        self._stdout_handle = None
        self._stderr_handle = None
        self._process: subprocess.Popen[str] | None = None
        self._spawned = False
        self._command = _build_llama_server_command(
            server_exe=self.server_exe,
            host=self.host,
            port=self.port,
            model_path=self.model_path,
            mmproj_path=self.mmproj_path,
            ctx_size=self.ctx_size,
            threads=threads,
            gpu_layers=str(gpu_layers),
            main_gpu=int(main_gpu),
            max_tokens=int(max_tokens),
            flash_attn=str(flash_attn),
            cache_type_k=self.cache_type_k,
            cache_type_v=self.cache_type_v,
            cache_prompt=self.cache_prompt,
            chat_lora_path=self.chat_lora_path,
        )
        if auto_start:
            self._start_process()

    @staticmethod
    def _resolve_prompt_model_path(prompt_model_path: Path | None) -> Path | None:
        if prompt_model_path is None:
            return None
        resolved = Path(prompt_model_path).expanduser().resolve()
        if not resolved.exists():
            return None
        if not (resolved / "chat_template.json").exists():
            return None
        return resolved

    @property
    def command(self) -> tuple[str, ...]:
        return tuple(self._command)

    @property
    def pid(self) -> int | None:
        if self._process is None or self._process.poll() is not None:
            return None
        return int(self._process.pid)

    def _cleanup_dead_process(self) -> None:
        if self._process is None or self._process.poll() is None:
            return
        self._process = None
        self._lora_adapters_cache = None
        self._chat_lora_id = None
        if self._stdout_handle is not None:
            self._stdout_handle.close()
            self._stdout_handle = None
        if self._stderr_handle is not None:
            self._stderr_handle.close()
            self._stderr_handle = None

    def _get_hf_prompt_processor(self):
        if self._hf_prompt_load_attempted:
            return self._hf_prompt_processor
        self._hf_prompt_load_attempted = True
        if self.prompt_model_path is None:
            return None
        try:
            from transformers import AutoProcessor
        except Exception as exc:  # noqa: BLE001
            self._hf_prompt_error = f"{type(exc).__name__}: {exc}"
            return None
        try:
            self._hf_prompt_processor = AutoProcessor.from_pretrained(
                str(self.prompt_model_path),
                local_files_only=True,
                use_fast=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._hf_prompt_error = f"{type(exc).__name__}: {exc}"
            self._hf_prompt_processor = None
        return self._hf_prompt_processor

    def _serialize_prompt_from_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        processor = self._get_hf_prompt_processor()
        if processor is None:
            return None
        return _serialize_hf_chat_template_prompt(processor, messages)

    def healthy(self, *, timeout_s: float = DEFAULT_HEALTH_POLL_TIMEOUT_S) -> bool:
        self._cleanup_dead_process()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=float(timeout_s))
            return response.status_code == 200
        except Exception:
            return False

    def ensure_running(self) -> None:
        if self.healthy(timeout_s=0.5):
            return
        self._start_process()
        deadline = time.monotonic() + self.health_timeout_s
        while time.monotonic() < deadline:
            if self.healthy(timeout_s=0.5):
                return
            if self._process is not None and self._process.poll() is not None:
                break
            time.sleep(0.5)
        raise RuntimeError(
            "Timed out waiting for llama.cpp sidecar readiness at "
            f"{self.base_url} using model={self.model_path} mmproj={self.mmproj_path} "
            f"(timeout={self.health_timeout_s:.1f}s, stderr_log={self.stderr_log}, stdout_log={self.stdout_log})"
        )

    def _start_process(self) -> None:
        self._cleanup_dead_process()
        if self._process is not None and self._process.poll() is None:
            return
        self.stdout_log.parent.mkdir(parents=True, exist_ok=True)
        self._lora_adapters_cache = None
        self._chat_lora_id = None
        self._stdout_handle = open(self.stdout_log, "w", encoding="utf-8")
        self._stderr_handle = open(self.stderr_log, "w", encoding="utf-8")
        env = os.environ.copy()
        server_dir = str(self.server_exe.parent)
        env["PATH"] = server_dir + os.pathsep + env.get("PATH", "")
        self._process = subprocess.Popen(
            self._command,
            cwd=str(self.root),
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
            env=env,
        )
        self._spawned = True

    def close(self) -> None:
        try:
            if self._spawned and self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
        finally:
            if self._stdout_handle is not None:
                self._stdout_handle.close()
                self._stdout_handle = None
            if self._stderr_handle is not None:
                self._stderr_handle.close()
                self._stderr_handle = None
            self._process = None
            self._lora_adapters_cache = None
            self._chat_lora_id = None

    def list_lora_adapters(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if self.chat_lora_path is None:
            return []
        if refresh or self._lora_adapters_cache is None:
            self.ensure_running()
            response = requests.get(f"{self.base_url}/lora-adapters", timeout=DEFAULT_REQUEST_TIMEOUT_S)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, list):
                raise RuntimeError(f"Expected JSON list from llama.cpp /lora-adapters, got {type(body)!r}.")
            self._lora_adapters_cache = [dict(item) for item in body if isinstance(item, dict)]
        return copy.deepcopy(self._lora_adapters_cache)

    def resolve_chat_lora_id(self, *, refresh: bool = False) -> int | None:
        if self.chat_lora_path is None:
            return None
        if self._chat_lora_id is not None and not refresh:
            return int(self._chat_lora_id)
        target = str(self.chat_lora_path).lower()
        target_name = self.chat_lora_path.name.lower()
        for adapter in self.list_lora_adapters(refresh=refresh):
            adapter_id = adapter.get("id")
            adapter_path = str(adapter.get("path", "")).strip()
            if adapter_id is None or not adapter_path:
                continue
            adapter_name = Path(adapter_path).name.lower()
            adapter_resolved = str(Path(adapter_path).expanduser()).lower()
            if adapter_name == target_name or adapter_resolved == target or adapter_path.lower() == target:
                self._chat_lora_id = int(adapter_id)
                return int(self._chat_lora_id)
        raise RuntimeError(
            "Configured chat LoRA adapter was not reported by llama.cpp /lora-adapters: "
            f"path={self.chat_lora_path}"
        )

    def chat_lora_request(self) -> list[dict[str, Any]]:
        if self.chat_lora_path is None:
            raise RuntimeError("Chat session LoRA is not configured.")
        return [{"id": int(self.resolve_chat_lora_id()), "scale": float(self.chat_lora_scale)}]

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        lora: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.ensure_running()
        serialized_prompt = self._serialize_prompt_from_messages(messages)
        if serialized_prompt is None or not self._hf_low_level_completions_enabled:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": int(max_tokens),
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
            }
            if lora:
                payload["lora"] = copy.deepcopy(lora)
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=DEFAULT_REQUEST_TIMEOUT_S,
            )
        else:
            payload = {
                "prompt": serialized_prompt,
                "n_predict": int(max_tokens),
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
            }
            if lora:
                payload["lora"] = copy.deepcopy(lora)
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    timeout=DEFAULT_REQUEST_TIMEOUT_S,
                )
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                response_obj = getattr(exc, "response", None)
                if response_obj is None or getattr(response_obj, "status_code", None) not in {400, 404, 405}:
                    raise
                self._hf_low_level_completions_enabled = False
                self._hf_prompt_error = (
                    "HF low-level /completions prompt serialization failed; "
                    "falling back to /v1/chat/completions for this server build."
                )
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": int(max_tokens),
                    "temperature": 0,
                    "top_k": 1,
                    "top_p": 1,
                }
                if lora:
                    payload["lora"] = copy.deepcopy(lora)
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=DEFAULT_REQUEST_TIMEOUT_S,
                )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"Expected JSON object from llama.cpp chat completion, got {type(body)!r}.")
        if "choices" not in body and "content" in body:
            body = {
                "choices": [{"text": _extract_text_content(body.get("content"))}],
                **body,
            }
        return body

    def health_payload(self) -> dict[str, Any]:
        payload = {
            "llama_url": self.base_url,
            "llama_cpp_root": str(self.root),
            "llama_server_path": str(self.server_exe),
            "llama_model_path": str(self.model_path),
            "llama_mmproj_path": str(self.mmproj_path),
            "llama_sidecar_ready": self.healthy(),
            "llama_sidecar_spawned": bool(self._spawned),
            "llama_sidecar_pid": self.pid,
            "llama_command": list(self.command),
            "llama_stdout_log": str(self.stdout_log),
            "llama_stderr_log": str(self.stderr_log),
            "llama_ctx_size": self.ctx_size,
            "llama_health_timeout_s": self.health_timeout_s,
            "embeddings": self.embeddings,
            "llama_cache_type_k": self.cache_type_k,
            "llama_cache_type_v": self.cache_type_v,
            "hf_prompt_serialization": self.prompt_model_path is not None,
            "hf_prompt_model_path": None if self.prompt_model_path is None else str(self.prompt_model_path),
            "hf_low_level_completions_enabled": self._hf_low_level_completions_enabled,
            "llama_chat_lora_path": None if self.chat_lora_path is None else str(self.chat_lora_path),
            "llama_chat_lora_scale": self.chat_lora_scale,
            "llama_chat_lora_id": self._chat_lora_id,
            "llama_chat_lora_configured": self.chat_lora_path is not None,
        }
        if self._hf_prompt_error:
            payload["hf_prompt_error"] = self._hf_prompt_error
        return payload
