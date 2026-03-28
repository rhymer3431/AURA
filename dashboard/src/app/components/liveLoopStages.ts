import type { DashboardState } from "../types";
import {
  architectureModule,
  architectureNode,
  asArray,
  asRecord,
  formatMeters,
  formatMs,
  formatRadians,
  stringValue,
} from "../selectors";

export type LoopStageId =
  | "gateway"
  | "perception"
  | "memory"
  | "s2"
  | "nav"
  | "locomotion"
  | "command-resolver";

export type LoopStage = {
  id: LoopStageId;
  label: string;
  status: string;
  summary: string;
  latency: string;
  output: string;
  details: Array<{ label: string; value: string }>;
};

function formatVector(value: unknown): string {
  if (!Array.isArray(value) || value.length < 3) {
    return "n/a";
  }
  const [vx, vy, wz] = value;
  if (
    typeof vx !== "number"
    || typeof vy !== "number"
    || typeof wz !== "number"
    || Number.isNaN(vx)
    || Number.isNaN(vy)
    || Number.isNaN(wz)
  ) {
    return "n/a";
  }
  return `[${vx.toFixed(2)}, ${vy.toFixed(2)}, ${wz.toFixed(2)}]`;
}

export function buildLoopStages(state: DashboardState | null): LoopStage[] {
  const runtime = asRecord(state?.runtime);
  const sensors = asRecord(state?.sensors);
  const memory = asRecord(state?.memory);
  const services = state?.services ?? {};
  const system2Output = state?.services.system2?.output ?? null;
  const selectedTarget = state?.selectedTargetSummary;
  const gateway = architectureNode(state, "gateway");
  const perception = architectureModule(state, "perception");
  const memoryModule = architectureModule(state, "memory");
  const s2 = architectureModule(state, "s2");
  const nav = architectureModule(state, "nav");
  const locomotion = architectureModule(state, "locomotion");
  const resolver = state?.architecture.mainControlServer.core.commandResolver ?? {
    name: "Command Resolver",
    status: "unknown",
    summary: "",
    detail: "",
    required: true,
    metrics: {},
  };
  const actionStatus = asRecord(runtime.actionStatus);
  const latency = state?.latencyBreakdown;

  return [
    {
      id: "gateway",
      label: "Gateway",
      status: gateway.status,
      summary: gateway.summary || "Frame ingress",
      latency: formatMs(latency?.frameAgeMs, "n/a"),
      output: `frame ${String(sensors.frameId ?? "n/a")} · ${stringValue(sensors.source, "unknown")}`,
      details: [
        { label: "frame id", value: String(sensors.frameId ?? "n/a") },
        { label: "source", value: stringValue(sensors.source, "n/a") },
        { label: "sensor ready", value: state?.session.active ? "yes" : "pending" },
        { label: "frame age", value: formatMs(latency?.frameAgeMs, "n/a") },
      ],
    },
    {
      id: "perception",
      label: "Perception",
      status: perception.status,
      summary: perception.summary || "Detection state",
      latency: formatMs(latency?.perceptionLatencyMs, "n/a"),
      output: `${String(state?.perception.detectionCount ?? 0)} detections · ${String(state?.perception.trackedDetectionCount ?? 0)} tracked`,
      details: [
        { label: "detector", value: stringValue(state?.perception.detectorBackend, "n/a") },
        { label: "selected reason", value: stringValue(state?.perception.detectorSelectedReason, "n/a") },
        { label: "target", value: selectedTarget?.className || selectedTarget?.trackId || "none" },
        { label: "trajectory points", value: String(state?.perception.trajectoryPointCount ?? 0) },
      ],
    },
    {
      id: "memory",
      label: "Memory",
      status: memoryModule.status,
      summary: memoryModule.summary || "Memory state",
      latency: formatMs(latency?.memoryLatencyMs, "n/a"),
      output: `${String(memory.objectCount ?? 0)} objects · ${String(memory.placeCount ?? 0)} places`,
      details: [
        { label: "memory-aware", value: memory.memoryAwareTaskActive ? "active" : "idle" },
        { label: "scratchpad", value: stringValue(asRecord(memory.scratchpad).taskState, "idle") },
        { label: "next priority", value: stringValue(asRecord(memory.scratchpad).nextPriority, "n/a") || "n/a" },
        { label: "retrieval", value: state?.selectedTargetSummary?.source ?? "pending" },
      ],
    },
    {
      id: "s2",
      label: "S2",
      status: s2.status,
      summary: s2.summary || "Decision path",
      latency: formatMs(latency?.s2LatencyMs ?? services.dual?.latencyMs, "n/a"),
      output: system2Output?.rawText || "no S2 output",
      details: [
        { label: "decision mode", value: system2Output?.decisionMode || "n/a" },
        { label: "needs requery", value: system2Output?.needsRequery ? "yes" : "no" },
        { label: "requested stop", value: system2Output?.requestedStop ? "yes" : "no" },
        {
          label: "pixel goal",
          value: Array.isArray(runtime.system2PixelGoal) ? runtime.system2PixelGoal.join(", ") : "n/a",
        },
      ],
    },
    {
      id: "nav",
      label: "Nav",
      status: nav.status,
      summary: nav.summary || "Route state",
      latency: formatMs(latency?.navLatencyMs, "n/a"),
      output: `plan v${String(runtime.planVersion ?? 0)} · traj v${String(runtime.trajVersion ?? 0)}`,
      details: [
        { label: "goal version", value: `v${String(runtime.goalVersion ?? 0)}` },
        { label: "goal distance", value: formatMeters(runtime.goalDistanceM, "n/a") },
        { label: "waypoint", value: `${String(runtime.globalRouteWaypointIndex ?? 0)} / ${String(runtime.globalRouteWaypointCount ?? 0)}` },
        { label: "route state", value: stringValue(runtime.plannerControlReason, "n/a") || "n/a" },
      ],
    },
    {
      id: "locomotion",
      label: "Locomotion",
      status: locomotion.status,
      summary: locomotion.summary || "Command vector",
      latency: formatMs(latency?.locomotionLatencyMs, "n/a"),
      output: formatVector(runtime.commandVector),
      details: [
        { label: "active command", value: stringValue(runtime.activeCommandType, "none") },
        { label: "command speed", value: formatMeters(runtime.commandSpeedMps, "n/a") },
        { label: "yaw error", value: formatRadians(runtime.yawErrorRad ?? runtime.plannerYawDeltaRad, "n/a") },
        { label: "trajectory points", value: String(runtime.navTrajectoryPointCount ?? asArray(runtime.navTrajectoryWorld).length ?? 0) },
      ],
    },
    {
      id: "command-resolver",
      label: "Command Resolver",
      status: resolver.status,
      summary: resolver.summary || "Action status",
      latency: "n/a",
      output: stringValue(actionStatus.reason, "clear") || "clear",
      details: [
        { label: "action state", value: stringValue(actionStatus.state, "n/a") || "n/a" },
        { label: "action reason", value: stringValue(actionStatus.reason, "clear") || "clear" },
        { label: "recovery state", value: stringValue(runtime.recoveryState, "NORMAL") },
        { label: "recovery reason", value: stringValue(runtime.recoveryReason, "clear") || "clear" },
      ],
    },
  ];
}
