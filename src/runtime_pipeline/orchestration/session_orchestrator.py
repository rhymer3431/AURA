from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    script_path: Path
    args: tuple[str, ...]
    health_url: str
    debug_url: str = ""
    tcp_ready_host: str | None = None
    tcp_ready_port: int | None = None
    env: tuple[tuple[str, str], ...] = ()


class SessionOrchestrator:
    def __init__(self, *, config, reserve_port: Callable[..., int]) -> None:  # noqa: ANN001
        self._config = config
        self._reserve_port = reserve_port

    def build_specs(self, request, *, scripts_dir_resolver: Callable[..., Path]) -> list[ProcessSpec]:  # noqa: ANN001
        scripts_dir = scripts_dir_resolver(self._config.repo_root, "scripts")
        navdp_port = self._reserve_port("127.0.0.1", 8888)
        system2_port = self._reserve_port("127.0.0.1", 15801, reserved={navdp_port})
        system2_sidecar_port = self._reserve_port(
            "127.0.0.1",
            system2_port + 1,
            reserved={navdp_port, system2_port},
        )
        navdp_base_url = f"http://127.0.0.1:{navdp_port}"
        system2_base_url = f"http://127.0.0.1:{system2_port}"
        system2_health_url = f"{system2_base_url}/healthz"
        runtime_args = self.runtime_args(
            request,
            navdp_base_url=navdp_base_url,
            system2_base_url=system2_base_url,
        )
        return [
            ProcessSpec(
                name="navdp",
                script_path=scripts_dir / "run_system.ps1",
                args=("-Component", "nav"),
                health_url=f"{navdp_base_url}/health",
                debug_url=f"{navdp_base_url}/debug_last_input",
                tcp_ready_host="127.0.0.1",
                tcp_ready_port=navdp_port,
                env=(("NAVDP_PORT", str(navdp_port)),),
            ),
            ProcessSpec(
                name="system2",
                script_path=scripts_dir / "run_system.ps1",
                args=("-Component", "s2"),
                health_url=system2_health_url,
                tcp_ready_host="127.0.0.1",
                tcp_ready_port=system2_port,
                env=(
                    ("INTERNVLA_HOST", "127.0.0.1"),
                    ("INTERNVLA_PORT", str(system2_port)),
                    ("INTERNVLA_LLAMA_URL", f"http://127.0.0.1:{system2_sidecar_port}"),
                ),
            ),
            ProcessSpec(
                name="runtime",
                script_path=scripts_dir / "run_system.ps1",
                args=("-Component", "runtime", *runtime_args),
                health_url="",
                env=(
                    ("G1_POINTGOAL_SCENE_PRESET", request.scene_preset),
                    ("AURA_RUNTIME_TRACE_PATH", str(self._config.process_log_dir / "runtime.trace.log")),
                ),
            ),
        ]

    @staticmethod
    def runtime_args(
        request,
        *,
        navdp_base_url: str,
        system2_base_url: str,
    ) -> list[str]:  # noqa: ANN001
        args = [
            "--native-viewer",
            "off",
            "--server-url",
            navdp_base_url,
            "--system2-url",
            system2_base_url,
        ]
        if request.launch_mode == "gui":
            args += ["--launch-mode", "gui"]
        else:
            args += ["--headless"]
        if request.viewer_enabled:
            args += ["--viewer-publish"]
        else:
            args += ["--no-viewer-publish"]
        if not request.memory_store:
            args += ["--no-memory-store"]
        if not request.detection_enabled:
            args += ["--skip-detection"]
        args += [
            "--action-scale",
            str(request.locomotion_config.action_scale),
            "--onnx-device",
            request.locomotion_config.onnx_device,
            "--cmd-max-vx",
            str(request.locomotion_config.cmd_max_vx),
            "--cmd-max-vy",
            str(request.locomotion_config.cmd_max_vy),
            "--cmd-max-wz",
            str(request.locomotion_config.cmd_max_wz),
        ]
        return args
