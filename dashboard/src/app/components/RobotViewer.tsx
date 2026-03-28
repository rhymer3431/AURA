import { useEffect, useRef, useState } from "react";
import { Eye, EyeOff, Maximize2, SignalHigh, Video, Zap } from "lucide-react";

import { useDashboard } from "../state";
import { useWebRTCViewer } from "../hooks/useWebRTCViewer";
import { asArray, asRecord, formatMs, numberValue, stringValue } from "../selectors";
import { buildApiUrl } from "../network";
import { ConsolePanel } from "./console-ui";

function cssVar(name: string, fallback: string) {
  if (typeof window === "undefined") {
    return fallback;
  }
  const value = window.getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value === "" ? fallback : value;
}

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

  const uiFont = cssVar("--font-ui", "sans-serif");
  const monoFont = cssVar("--font-mono", "monospace");
  const detectionColor = cssVar("--signal-emerald", "#7d8f7d");
  const trajectoryColor = cssVar("--signal-cyan", "#82939a");
  const navGoalColor = cssVar("--signal-coral", "#a2776c");
  const systemGoalColor = cssVar("--signal-amber", "#a18a69");

  const detections = asArray<Record<string, unknown>>(telemetry?.detections);
  if (detections.length > 0) {
    context.lineWidth = 2;
    context.strokeStyle = detectionColor;
    context.fillStyle = detectionColor;
    context.font = `12px ${uiFont}`;
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
    context.strokeStyle = trajectoryColor;
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
    context.strokeStyle = navGoalColor;
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
    context.strokeStyle = systemGoalColor;
    context.fillStyle = systemGoalColor;
    context.font = `11px ${monoFont}`;
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
  const viewerStateLabel = viewer.connected ? "LIVE" : viewerEnabled ? "CONNECTING" : "OFFLINE";
  const trackLabel = trackRoles.length > 0 ? trackRoles.join(", ") : "none";

  return (
    <ConsolePanel className="h-full flex flex-col">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <Video className="size-4 text-[var(--text-tertiary)]" />
          <h3 className="text-[15px] font-semibold text-[var(--foreground)]">Live Robot View</h3>
          <span className="dashboard-live-pill">
            <span className="dashboard-live-pill-dot" />
            {viewerStateLabel}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOverlay((current) => !current)}
            className={`dashboard-button-secondary !rounded-[12px] !px-3 !py-1.5 text-[11px] ${
              showOverlay
                ? "border-[var(--tone-cyan-border)] bg-[var(--surface-2)] text-[var(--foreground)]"
                : "border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-strong)] text-[var(--text-secondary)]"
            }`}
          >
            {showOverlay ? <Eye className="size-3.5" /> : <EyeOff className="size-3.5" />}
            BBox Overlays
          </button>
          <button type="button" aria-label="Expand viewer" className="dashboard-utility-button">
            <Maximize2 className="size-[14px]" />
          </button>
        </div>
      </div>

      <div className="relative w-full aspect-video overflow-hidden rounded-[18px] border border-[rgba(var(--ink-rgb),0.08)] bg-[#0f1720]">
        <video
          ref={viewer.rgbVideoRef}
          className="w-full h-full object-cover"
          autoPlay
          muted
          playsInline
        />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />

        <div className="pointer-events-none absolute left-0 top-0 flex w-full items-start justify-between p-2.5">
          <div className="flex flex-col gap-1">
            <div className="dashboard-viewer-hud">
              CAM: {frameSource}
            </div>
            <div className="dashboard-viewer-hud">
              FPS: {viewer.connected ? "30" : "0"} | RES: {numberValue(image.width) ?? 0}x{numberValue(image.height) ?? 0}
            </div>
          </div>
          <div className="dashboard-viewer-hud flex items-center gap-1.5">
            <SignalHigh className="size-3 text-[var(--signal-emerald)]" />
            frame age {formatMs(state?.transport.frameAgeMs, "n/a")}
          </div>
        </div>

        {!viewerEnabled && (
          <div className="absolute inset-0 flex items-center justify-center bg-[rgba(var(--ink-rgb),0.52)] text-[rgba(var(--paper-rgb),0.9)] text-[13px]">
            viewer publish disabled
          </div>
        )}
        {viewerEnabled && waitingForFrame && (
          <div className="absolute inset-0 flex items-center justify-center bg-[rgba(var(--ink-rgb),0.42)] text-[rgba(var(--paper-rgb),0.9)] text-[13px]">
            waiting for frame
          </div>
        )}
        {viewer.error !== "" && (
          <div className="absolute right-3 bottom-3 left-3 rounded-lg border border-[var(--tone-coral-border)] bg-[var(--tone-coral-bg)] px-3 py-2 text-[11px] text-[var(--tone-coral-fg)]">
            {viewer.error}
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 px-1">
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-[var(--text-secondary)]">
          <div className="flex items-center gap-1.5">
            <Zap className="size-3.5 text-[var(--signal-emerald)]" />
            <span>
              Inference: <span className="font-medium text-[var(--foreground)]">{detectorBackend}</span>
            </span>
          </div>
          <div className="dashboard-inline-divider" />
          <span>
            Detected: <span className="font-medium text-[var(--foreground)]">{detections.length} objects</span>
          </span>
          <div className="dashboard-inline-divider" />
          <span>
            Tracks: <span className="font-medium text-[var(--foreground)]">{trackLabel}</span>
          </span>
        </div>
        <div className="dashboard-mono text-[10px] text-[var(--text-tertiary)]">
          peer {peerSessionId}
        </div>
      </div>
    </ConsolePanel>
  );
}
