import { useEffect, useRef, useState } from "react";
import { Maximize2, Eye, EyeOff, Video, SignalHigh } from "lucide-react";

import { useDashboard } from "../state";
import { useWebRTCViewer } from "../hooks/useWebRTCViewer";
import { asArray, asRecord, formatMs, numberValue, stringValue } from "../selectors";
import { buildApiUrl } from "../network";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle } from "./console-ui";

function drawOverlay(
  canvas: HTMLCanvasElement,
  snapshot: Record<string, unknown> | null,
  telemetry: Record<string, unknown> | null,
) {
  const context = canvas.getContext("2d");
  if (context === null) {
    return;
  }
  const image = asRecord(snapshot?.image);
  const sourceWidth = numberValue(image.width) ?? 1;
  const sourceHeight = numberValue(image.height) ?? 1;
  const width = Math.max(canvas.clientWidth, 1);
  const height = Math.max(canvas.clientHeight, 1);
  if (canvas.width !== width) {
    canvas.width = width;
  }
  if (canvas.height !== height) {
    canvas.height = height;
  }
  context.clearRect(0, 0, width, height);

  const detections = asArray<Record<string, unknown>>(telemetry?.detections);
  if (detections.length > 0) {
    context.lineWidth = 2;
    context.strokeStyle = "#10b981";
    context.fillStyle = "#10b981";
    context.font = "12px sans-serif";
    detections.forEach((item) => {
      const bbox = asArray<number>(item.bbox_xyxy);
      if (bbox.length !== 4) {
        return;
      }
      const x = (bbox[0] / sourceWidth) * width;
      const y = (bbox[1] / sourceHeight) * height;
      const boxWidth = ((bbox[2] - bbox[0]) / sourceWidth) * width;
      const boxHeight = ((bbox[3] - bbox[1]) / sourceHeight) * height;
      context.strokeRect(x, y, boxWidth, boxHeight);
      const label = stringValue(item.class_name, "object");
      const confidence = numberValue(item.confidence);
      context.fillText(
        confidence === null ? label : `${label} ${Math.round(confidence * 100)}%`,
        x + 4,
        Math.max(y - 6, 12),
      );
    });
  }

  const trajectory = asArray<Array<number>>(telemetry?.trajectoryPixels ?? telemetry?.trajectory_pixels);
  if (trajectory.length > 1) {
    context.beginPath();
    context.lineWidth = 2;
    context.strokeStyle = "#38bdf8";
    trajectory.forEach((point, index) => {
      if (!Array.isArray(point) || point.length !== 2) {
        return;
      }
      const x = (Number(point[0]) / sourceWidth) * width;
      const y = (Number(point[1]) / sourceHeight) * height;
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  }

  const activeTarget = asRecord(telemetry?.activeTarget ?? telemetry?.active_target ?? snapshot?.activeTarget ?? snapshot?.active_target);
  const navGoalPixel = asArray<number>(activeTarget.nav_goal_pixel);
  if (navGoalPixel.length === 2) {
    const x = (Number(navGoalPixel[0]) / sourceWidth) * width;
    const y = (Number(navGoalPixel[1]) / sourceHeight) * height;
    context.strokeStyle = "#f97316";
    context.beginPath();
    context.arc(x, y, 8, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.moveTo(x - 10, y);
    context.lineTo(x + 10, y);
    context.moveTo(x, y - 10);
    context.lineTo(x, y + 10);
    context.stroke();
  }

  const system2PixelGoal = asArray<number>(
    telemetry?.system2PixelGoal ?? telemetry?.system2_pixel_goal ?? snapshot?.system2PixelGoal ?? snapshot?.system2_pixel_goal,
  );
  if (system2PixelGoal.length === 2) {
    const x = (Number(system2PixelGoal[0]) / sourceWidth) * width;
    const y = (Number(system2PixelGoal[1]) / sourceHeight) * height;
    context.save();
    context.lineWidth = 2;
    context.strokeStyle = "#facc15";
    context.fillStyle = "#facc15";
    context.font = "11px monospace";
    context.beginPath();
    context.arc(x, y, 10, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.moveTo(x - 14, y);
    context.lineTo(x + 14, y);
    context.moveTo(x, y - 14);
    context.lineTo(x, y + 14);
    context.stroke();
    context.fillText("S2 GOAL", Math.min(x + 14, width - 56), Math.max(y - 12, 14));
    context.restore();
  }
}

export function RobotViewer() {
  const { bootstrap, state } = useDashboard();
  const [showOverlay, setShowOverlay] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const viewerEnabled = state?.session.active === true && Boolean(state?.transport.viewerEnabled);
  const viewer = useWebRTCViewer({
    basePath: bootstrap?.webrtcBasePath ?? buildApiUrl("/api/webrtc"),
    enabled: viewerEnabled,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return undefined;
    }
    if (!showOverlay) {
      const context = canvas.getContext("2d");
      context?.clearRect(0, 0, canvas.width, canvas.height);
      return undefined;
    }
    let frameId = 0;
    const render = () => {
      drawOverlay(
        canvas,
        (viewer.snapshotRef.current as Record<string, unknown> | null) ?? null,
        (viewer.telemetryRef.current as Record<string, unknown> | null) ?? null,
      );
      frameId = window.requestAnimationFrame(render);
    };
    render();
    return () => window.cancelAnimationFrame(frameId);
  }, [showOverlay, viewer.snapshotRef, viewer.telemetryRef]);

  const snapshot = asRecord(viewer.snapshotRef.current);
  const telemetry = asRecord(viewer.telemetryRef.current);
  const image = asRecord(snapshot.image);
  const trackRoles = viewer.trackRoles.length > 0 ? viewer.trackRoles : asArray<string>(state?.transport.peerTrackRoles);
  const detections = asArray<Record<string, unknown>>(telemetry.detections);
  const waitingForFrame = stringValue(snapshot.type) === "waiting_for_frame";
  const frameSource = stringValue(snapshot.source, stringValue(state?.sensors.source, "aura_runtime"));
  const detectorBackend = stringValue(snapshot.detector_backend, stringValue(state?.perception.detectorBackend, "unknown"));
  const peerSessionId = stringValue(state?.transport.peerSessionId, "none");

  return (
    <ConsolePanel className="h-full flex flex-col">
      <div className="mb-4 flex items-start justify-between gap-4">
        <ConsoleSectionTitle
          icon={Video}
          eyebrow="vision feed"
          title="Live Robot View"
          description="RGB stream, detection overlays, trajectory trace, and active target markers"
        />

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOverlay((current) => !current)}
            className={`dashboard-button-secondary !rounded-full !px-3 !py-2 text-[11px] ${
              showOverlay
                ? "border-[rgba(79,152,168,0.16)] bg-[rgba(79,152,168,0.1)] text-[var(--tone-cyan-fg)]"
                : "text-[var(--text-secondary)]"
            }`}
          >
            {showOverlay ? <Eye className="size-3.5" /> : <EyeOff className="size-3.5" />}
            BBox Overlays
          </button>
          <button className="dashboard-button-secondary !rounded-full !px-3 !py-2 text-[var(--text-secondary)]">
            <Maximize2 className="size-3.5" />
          </button>
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <ConsoleBadge tone={viewer.connected ? "emerald" : "amber"}>
          {viewer.connected ? "WEBRTC" : viewerEnabled ? "CONNECTING" : "INACTIVE"}
        </ConsoleBadge>
        <ConsoleBadge tone="slate" dot={false}>
          source {frameSource}
        </ConsoleBadge>
        <ConsoleBadge tone="cyan" dot={false}>
          detector {detectorBackend}
        </ConsoleBadge>
      </div>

      <div className="relative w-full aspect-video overflow-hidden rounded-[20px] border border-[rgba(17,23,28,0.08)] bg-neutral-950">
        <video
          ref={viewer.rgbVideoRef}
          className="w-full h-full object-cover"
          autoPlay
          muted
          playsInline
        />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />

        <div className="absolute top-0 left-0 w-full p-3 flex justify-between items-start pointer-events-none">
          <div className="flex flex-col gap-1">
            <div className="dashboard-mono text-[10px] text-white/80 bg-black/45 px-2.5 py-1 rounded-full backdrop-blur-sm">
              SRC: {frameSource}
            </div>
            <div className="dashboard-mono text-[10px] text-white/80 bg-black/45 px-2.5 py-1 rounded-full backdrop-blur-sm">
              RES: {numberValue(image.width) ?? 0}x{numberValue(image.height) ?? 0} | TRACKS: {trackRoles.join(",") || "none"}
            </div>
          </div>
          <div className="dashboard-mono flex items-center gap-1.5 text-[10px] text-white/80 bg-black/45 px-2.5 py-1 rounded-full backdrop-blur-sm">
            <SignalHigh className="size-3 text-emerald-400" />
            frame age {formatMs(state?.transport.frameAgeMs, "n/a")}
          </div>
        </div>

        {!viewerEnabled && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/45 text-white text-[13px]">
            viewer publish disabled
          </div>
        )}
        {viewerEnabled && waitingForFrame && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/35 text-white text-[13px]">
            waiting for frame
          </div>
        )}
        {viewer.error !== "" && (
          <div className="absolute left-3 right-3 bottom-3 rounded-lg bg-red-500/90 px-3 py-2 text-[11px] text-white">
            {viewer.error}
          </div>
        )}
      </div>

      <div className="dashboard-inset mt-4 px-4 py-3">
        <div className="dashboard-eyebrow mb-3">telemetry strip</div>
        <div className="flex items-center gap-4 flex-wrap">
          <div className="dashboard-micro">
            Inference: <span className="text-[var(--foreground)]">{detectorBackend}</span>
          </div>
          <div className="h-3 w-px bg-[rgba(24,33,37,0.08)]" />
          <div className="dashboard-micro">
            Detected: <span className="text-[var(--foreground)]">{detections.length} objects</span>
          </div>
          <div className="h-3 w-px bg-[rgba(24,33,37,0.08)]" />
          <div className="dashboard-micro">
            Trajectory: <span className="text-[var(--foreground)]">{asArray(telemetry.trajectoryPixels ?? telemetry.trajectory_pixels).length} pts</span>
          </div>
          <div className="ml-auto dashboard-micro">peer {peerSessionId}</div>
        </div>
      </div>
    </ConsolePanel>
  );
}
