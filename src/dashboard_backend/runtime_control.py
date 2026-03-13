from __future__ import annotations

from adapters.sensors.isaac_bridge_adapter import IsaacBridgeAdapter
from ipc.messages import RuntimeControlRequest, TaskRequest
from ipc.zmq_bus import ZmqBus


class RuntimeControlClient:
    def __init__(
        self,
        *,
        control_endpoint: str,
        telemetry_endpoint: str,
        identity: str = "dashboard_backend",
    ) -> None:
        self._bus = ZmqBus(
            control_endpoint=str(control_endpoint),
            telemetry_endpoint=str(telemetry_endpoint),
            role="agent",
            identity=str(identity),
        )
        self._bridge = IsaacBridgeAdapter(self._bus)

    def submit_task(self, instruction: str) -> TaskRequest:
        request = TaskRequest(command_text=str(instruction).strip())
        self._bridge.publish_task_request(request)
        return request

    def cancel_interactive_task(self) -> RuntimeControlRequest:
        request = RuntimeControlRequest(action="cancel_interactive_task")
        self._bridge.publish_runtime_control(request)
        return request

    def transport_health_snapshot(self) -> dict[str, object]:
        return self._bus.health().snapshot()

    def close(self) -> None:
        self._bus.close()
