"""Configuration for the managed inference stack."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(slots=True)
class ManagedServiceConfig:
    name: str
    host: str
    port: int
    health_path: str
    command: list[str]
    required: bool = True

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"


def _python_command() -> str:
    return sys.executable or "python"


def build_managed_services(args) -> list[ManagedServiceConfig]:
    python = _python_command()
    navdp_command = [
        python,
        "-m",
        "systems.inference.navdp.server",
        "--port",
        str(int(args.navdp_port)),
        "--checkpoint",
        str(args.navdp_checkpoint),
        "--device",
        str(args.navdp_device),
    ]
    system2_command = [
        python,
        "-m",
        "systems.inference.system2.server",
        "--host",
        str(args.system2_host),
        "--port",
        str(int(args.system2_port)),
        "--llama-url",
        str(args.system2_llama_url),
    ]
    if str(args.system2_model_path).strip():
        system2_command.extend(["--model-path", str(args.system2_model_path)])
    planner_command = [
        python,
        "-m",
        "systems.inference.planner.server",
        "--host",
        str(args.planner_host),
        "--port",
        str(int(args.planner_port)),
        "--model",
        str(args.planner_model_path),
        "--llama-server",
        str(args.planner_llama_server),
        "--gpu-layers",
        str(int(args.planner_gpu_layers)),
        "--ctx-size",
        str(int(args.planner_ctx_size)),
        "--cache-type-k",
        str(args.planner_cache_type_k),
        "--cache-type-v",
        str(args.planner_cache_type_v),
    ]
    return [
        ManagedServiceConfig(
            name="navdp",
            host=str(args.navdp_host),
            port=int(args.navdp_port),
            health_path="/healthz",
            command=navdp_command,
            required=False,
        ),
        ManagedServiceConfig(
            name="system2",
            host=str(args.system2_host),
            port=int(args.system2_port),
            health_path="/healthz",
            command=system2_command,
        ),
        ManagedServiceConfig(
            name="planner",
            host=str(args.planner_host),
            port=int(args.planner_port),
            health_path="/health",
            command=planner_command,
            required=False,
        ),
    ]


def default_log_dir() -> Path:
    return Path(os.environ.get("AURA_INFERENCE_STACK_LOG_DIR", REPO_ROOT / "logs" / "inference_stack"))
