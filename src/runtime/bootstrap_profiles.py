from __future__ import annotations

from dataclasses import dataclass, field


PROFILE_AUTO = "auto"
PROFILE_MINIMAL_HEADLESS_SENSOR = "minimal_headless_sensor_smoke"
PROFILE_STANDALONE_RENDER_WARMUP = "standalone_render_warmup"
PROFILE_EDITOR_ASSISTED = "full_app_editor_assisted"
PROFILE_EXTENSION_IN_EDITOR = "extension_in_editor"


@dataclass(frozen=True)
class BootstrapProfile:
    name: str
    description: str
    preferred_launch_mode: str
    required_experience: list[str] = field(default_factory=list)
    required_extensions: list[str] = field(default_factory=list)
    headless: bool | None = None
    render_warmup_updates: int = 0
    physics_warmup_steps: int = 0
    stage_settle_updates: int = 0
    sensor_init_retries: int = 0
    sensor_init_retry_updates: int = 0
    first_frame_timeout_sec: float = 20.0
    smoke_target_tier: str = "sensor"
    allow_empty_detections: bool = True
    require_nonempty_frame: bool = False


@dataclass(frozen=True)
class BootstrapProfileSelection:
    requested_profile: str
    selected_profile: BootstrapProfile
    reason: str


def profile_registry() -> dict[str, BootstrapProfile]:
    return {
        PROFILE_MINIMAL_HEADLESS_SENSOR: BootstrapProfile(
            name=PROFILE_MINIMAL_HEADLESS_SENSOR,
            description="Minimal standalone headless sensor validation with short warmup and sensor-tier target.",
            preferred_launch_mode="standalone_python",
            required_experience=["apps/isaacsim.exp.base.python.kit", "apps/isaacsim.exp.full.kit"],
            required_extensions=["isaacsim.sensors.camera"],
            headless=True,
            render_warmup_updates=6,
            physics_warmup_steps=0,
            stage_settle_updates=6,
            sensor_init_retries=2,
            sensor_init_retry_updates=4,
            first_frame_timeout_sec=18.0,
            smoke_target_tier="sensor",
            allow_empty_detections=True,
            require_nonempty_frame=False,
        ),
        PROFILE_STANDALONE_RENDER_WARMUP: BootstrapProfile(
            name=PROFILE_STANDALONE_RENDER_WARMUP,
            description="Standalone headless profile with heavier render warmup for RGB/depth/pipeline validation.",
            preferred_launch_mode="standalone_python",
            required_experience=["apps/isaacsim.exp.full.kit", "apps/isaacsim.exp.base.python.kit"],
            required_extensions=["isaacsim.sensors.camera"],
            headless=True,
            render_warmup_updates=18,
            physics_warmup_steps=6,
            stage_settle_updates=10,
            sensor_init_retries=3,
            sensor_init_retry_updates=6,
            first_frame_timeout_sec=30.0,
            smoke_target_tier="memory",
            allow_empty_detections=True,
            require_nonempty_frame=False,
        ),
        PROFILE_EDITOR_ASSISTED: BootstrapProfile(
            name=PROFILE_EDITOR_ASSISTED,
            description="In-editor profile that assumes an active Full App stage and favors pipeline/memory validation.",
            preferred_launch_mode="editor_assisted",
            required_experience=["apps/isaacsim.exp.full.kit"],
            required_extensions=["isaacsim.sensors.camera"],
            headless=False,
            render_warmup_updates=10,
            physics_warmup_steps=4,
            stage_settle_updates=6,
            sensor_init_retries=2,
            sensor_init_retry_updates=4,
            first_frame_timeout_sec=24.0,
            smoke_target_tier="memory",
            allow_empty_detections=True,
            require_nonempty_frame=False,
        ),
        PROFILE_EXTENSION_IN_EDITOR: BootstrapProfile(
            name=PROFILE_EXTENSION_IN_EDITOR,
            description="Extension-hosted in-editor profile for hot-reload debugging and action-driven smoke runs.",
            preferred_launch_mode="extension_mode",
            required_experience=["apps/isaacsim.exp.full.kit"],
            required_extensions=["isaacsim.sensors.camera"],
            headless=False,
            render_warmup_updates=12,
            physics_warmup_steps=4,
            stage_settle_updates=6,
            sensor_init_retries=2,
            sensor_init_retry_updates=4,
            first_frame_timeout_sec=24.0,
            smoke_target_tier="memory",
            allow_empty_detections=True,
            require_nonempty_frame=False,
        ),
    }


def list_profile_names() -> list[str]:
    return [PROFILE_AUTO, *profile_registry().keys()]


def select_bootstrap_profile(
    requested_profile: str,
    *,
    launch_mode: str,
    headless: bool,
    smoke_target_tier: str,
) -> BootstrapProfileSelection:
    registry = profile_registry()
    normalized_request = str(requested_profile).strip().lower() or PROFILE_AUTO
    normalized_launch = str(launch_mode).strip().lower()
    normalized_tier = str(smoke_target_tier).strip().lower() or "sensor"
    if normalized_request in registry:
        return BootstrapProfileSelection(
            requested_profile=normalized_request,
            selected_profile=registry[normalized_request],
            reason="bootstrap profile selected explicitly",
        )
    if normalized_launch == "extension_mode":
        return BootstrapProfileSelection(
            requested_profile=normalized_request,
            selected_profile=registry[PROFILE_EXTENSION_IN_EDITOR],
            reason="extension_mode requires the extension_in_editor bootstrap profile",
        )
    if normalized_launch == "editor_assisted":
        return BootstrapProfileSelection(
            requested_profile=normalized_request,
            selected_profile=registry[PROFILE_EDITOR_ASSISTED],
            reason="editor-assisted smoke prefers the in-editor bootstrap profile",
        )
    if normalized_tier == "sensor" and bool(headless):
        return BootstrapProfileSelection(
            requested_profile=normalized_request,
            selected_profile=registry[PROFILE_MINIMAL_HEADLESS_SENSOR],
            reason="headless sensor-tier smoke uses the minimal standalone profile",
        )
    return BootstrapProfileSelection(
        requested_profile=normalized_request,
        selected_profile=registry[PROFILE_STANDALONE_RENDER_WARMUP],
        reason="standalone smoke defaults to the render-warmup profile for higher frame-ingress success",
    )
