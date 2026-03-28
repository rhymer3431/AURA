import { useEffect, useMemo, useRef, useState } from "react";
import {
  Eye,
  EyeOff,
  Focus,
  Maximize2,
  SignalHigh,
  Video,
  Zap,
} from "lucide-react";

import { useDashboard } from "../state";
import { useWebRTCViewer } from "../hooks/useWebRTCViewer";
import { asArray, asRecord, formatMs, numberValue, stringValue } from "../selectors";
import { buildApiUrl } from "../network";
import { ConsoleBadge, ConsolePanel } from "./console-ui";

type ViewerDetection = {
  key: string;
  trackId: string;
  className: string;
  confidence: number | null;
  depthM: number | null;
  bbox: number[];
  worldPose: number[] | null;
};

type FilmstripFrame = {
  frameId: number;
  imageData: string | null;
  source: string;
  hasDecision: boolean;
};

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
  selectedTrackId: string | null,
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
  const selectedColor = cssVar("--signal-coral", "#a2776c");
  const trajectoryColor = cssVar("--signal-cyan", "#82939a");
  const navGoalColor = cssVar("--signal-coral", "#a2776c");
  const systemGoalColor = cssVar("--signal-amber", "#a18a69");

  const detections = asArray<Record<string, unknown>>(telemetry?.detections);
  if (detections.length > 0) {
    context.font = `12px ${uiFont}`;
    detections.forEach((item, index) => {
      const bbox = asArray<number>(item.bbox_xyxy);
      if (bbox.length !== 4) {
        return;
      }
      const trackId = stringValue(item.track_id, `detection-${index}`);
      const isSelected = selectedTrackId !== null && trackId === selectedTrackId;
      const x = (bbox[0] / sourceWidth) * width;
      const y = (bbox[1] / sourceHeight) * height;
      const boxWidth = ((bbox[2] - bbox[0]) / sourceWidth) * width;
      const boxHeight = ((bbox[3] - bbox[1]) / sourceHeight) * height;
      context.lineWidth = isSelected ? 3 : 2;
      context.strokeStyle = isSelected ? selectedColor : detectionColor;
      context.fillStyle = isSelected ? selectedColor : detectionColor;
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

function captureFilmstripFrame(video: HTMLVideoElement | null): string | null {
  if (video === null || video.videoWidth <= 0 || video.videoHeight <= 0) {
    return null;
  }
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(Math.round(video.videoWidth / 4), 1);
  canvas.height = Math.max(Math.round(video.videoHeight / 4), 1);
  const context = canvas.getContext("2d");
  if (context === null) {
    return null;
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.75);
}

export function RobotViewer({
  selectedTrackId = null,
  onSelectTrackId,
  selectedFrameId = null,
  onSelectFrameId,
}: {
  selectedTrackId?: string | null;
  onSelectTrackId?: (trackId: string | null) => void;
  selectedFrameId?: number | null;
  onSelectFrameId?: (frameId: number | null) => void;
}) {
  const { bootstrap, state } = useDashboard();
  const [showOverlay, setShowOverlay] = useState(true);
  const [filmstrip, setFilmstrip] = useState<FilmstripFrame[]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const viewerEnabled = state?.session.active === true && Boolean(state?.transport.viewerEnabled);
  const viewer = useWebRTCViewer({
    basePath: bootstrap?.webrtcBasePath ?? buildApiUrl("/api/webrtc"),
    enabled: viewerEnabled,
  });

  const snapshot = asRecord(viewer.snapshotRef.current);
  const telemetry = asRecord(viewer.telemetryRef.current);
  const image = asRecord(snapshot.image);
  const frameId = numberValue(snapshot.frame_id ?? snapshot.frameId);
  const detections = useMemo<ViewerDetection[]>(() => {
    return asArray<Record<string, unknown>>(telemetry.detections).map((item, index) => ({
      key: stringValue(item.track_id, `frame-${frameId ?? -1}-detection-${index}`),
      trackId: stringValue(item.track_id, ""),
      className: stringValue(item.class_name, "object"),
      confidence: numberValue(item.confidence),
      depthM: numberValue(item.depth_m),
      bbox: asArray<number>(item.bbox_xyxy),
      worldPose: asArray<number>(item.world_pose_xyz),
    }));
  }, [frameId, telemetry.detections]);
  const selectedTargetSummary = state?.selectedTargetSummary;
  const effectiveSelectedTrackId = selectedTrackId ?? selectedTargetSummary?.trackId ?? null;
  const selectedDetection =
    detections.find((item) => effectiveSelectedTrackId !== null && item.key === effectiveSelectedTrackId)
    ?? detections.find((item) => selectedTargetSummary?.trackId !== "" && item.trackId === selectedTargetSummary?.trackId)
    ?? null;
  const waitingForFrame = stringValue(snapshot.type) === "waiting_for_frame";
  const frameSource = stringValue(snapshot.source, stringValue(state?.sensors.source, "aura_runtime"));
  const detectorBackend = stringValue(snapshot.detector_backend, stringValue(state?.perception.detectorBackend, "unknown"));
  const peerSessionId = stringValue(state?.transport.peerSessionId, "none");
  const viewerStateLabel = viewer.connected ? "LIVE" : viewerEnabled ? "CONNECTING" : "OFFLINE";
  const trackRoles = viewer.trackRoles.length > 0 ? viewer.trackRoles : asArray<string>(state?.transport.peerTrackRoles);
  const trackLabel = trackRoles.length > 0 ? trackRoles.join(", ") : "none";
  const decisionFrames = new Set(
    (state?.cognitionTrace ?? [])
      .filter((item) => item.s2DecisionMode !== "")
      .map((item) => item.frameId),
  );
  const activeFilmstripFrameId = selectedFrameId ?? frameId ?? filmstrip[0]?.frameId ?? null;

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
    let animationFrame = 0;
    const render = () => {
      drawOverlay(canvas, snapshot, telemetry, effectiveSelectedTrackId);
      animationFrame = window.requestAnimationFrame(render);
    };
    render();
    return () => window.cancelAnimationFrame(animationFrame);
  }, [effectiveSelectedTrackId, showOverlay, snapshot, telemetry]);

  useEffect(() => {
    if (frameId === null) {
      return;
    }
    setFilmstrip((current) => {
      if (current.some((item) => item.frameId === frameId)) {
        return current;
      }
      const nextFrame: FilmstripFrame = {
        frameId,
        imageData: captureFilmstripFrame(viewer.rgbVideoRef.current),
        source: frameSource,
        hasDecision: decisionFrames.has(frameId),
      };
      return [nextFrame, ...current].slice(0, 6);
    });
  }, [decisionFrames, frameId, frameSource, viewer.hudVersion, viewer.rgbVideoRef]);

  useEffect(() => {
    if (effectiveSelectedTrackId === null) {
      return;
    }
    const exists = detections.some((item) => item.key === effectiveSelectedTrackId);
    if (!exists) {
      onSelectTrackId?.(null);
    }
  }, [detections, effectiveSelectedTrackId, onSelectTrackId]);

  useEffect(() => {
    if (selectedFrameId === null) {
      return;
    }
    const exists = filmstrip.some((item) => item.frameId === selectedFrameId);
    if (!exists) {
      onSelectFrameId?.(null);
    }
  }, [filmstrip, onSelectFrameId, selectedFrameId]);

  return (
    <ConsolePanel className="flex h-full flex-col gap-4">
      <div className="flex items-start justify-between gap-3">
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
            type="button"
            onClick={() => setShowOverlay((current) => !current)}
            className={`dashboard-button-secondary !rounded-[12px] !px-3 !py-1.5 text-[11px] ${
              showOverlay
                ? "border-[var(--tone-cyan-border)] bg-[var(--surface-2)] text-[var(--foreground)]"
                : "border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-strong)] text-[var(--text-secondary)]"
            }`}
          >
            {showOverlay ? <Eye className="size-3.5" /> : <EyeOff className="size-3.5" />}
            Overlay
          </button>
          <button type="button" aria-label="Expand viewer" className="dashboard-utility-button">
            <Maximize2 className="size-[14px]" />
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <ConsoleBadge tone="emerald">detections</ConsoleBadge>
        <ConsoleBadge tone="coral">active target</ConsoleBadge>
        <ConsoleBadge tone="amber">S2 goal</ConsoleBadge>
        <ConsoleBadge tone="cyan">trajectory</ConsoleBadge>
      </div>

      <div className="relative aspect-video w-full overflow-hidden rounded-[18px] border border-[rgba(var(--ink-rgb),0.08)] bg-[#0f1720]">
        <video ref={viewer.rgbVideoRef} className="h-full w-full object-cover" autoPlay muted playsInline />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />

        <div className="pointer-events-none absolute left-0 top-0 flex w-full items-start justify-between p-2.5">
          <div className="flex flex-col gap-1">
            <div className="dashboard-viewer-hud">CAM: {frameSource}</div>
            <div className="dashboard-viewer-hud">
              FPS: {viewer.connected ? "30" : "0"} | RES: {numberValue(image.width) ?? 0}x{numberValue(image.height) ?? 0}
            </div>
          </div>
          <div className="dashboard-viewer-hud flex items-center gap-1.5">
            <SignalHigh className="size-3 text-[var(--signal-emerald)]" />
            frame age {formatMs(state?.latencyBreakdown.frameAgeMs, "n/a")}
          </div>
        </div>

        {!viewerEnabled && (
          <div className="absolute inset-0 flex items-center justify-center bg-[rgba(var(--ink-rgb),0.52)] text-[13px] text-[rgba(var(--paper-rgb),0.9)]">
            viewer publish disabled
          </div>
        )}
        {viewerEnabled && waitingForFrame && (
          <div className="absolute inset-0 flex items-center justify-center bg-[rgba(var(--ink-rgb),0.42)] text-[13px] text-[rgba(var(--paper-rgb),0.9)]">
            waiting for frame
          </div>
        )}
        {viewer.error !== "" && (
          <div className="absolute bottom-3 left-3 right-3 rounded-lg border border-[var(--tone-coral-border)] bg-[var(--tone-coral-bg)] px-3 py-2 text-[11px] text-[var(--tone-coral-fg)]">
            {viewer.error}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(260px,0.9fr)]">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3 px-1">
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
            <div className="dashboard-mono text-[10px] text-[var(--text-tertiary)]">peer {peerSessionId}</div>
          </div>

          <div className="grid grid-cols-1 gap-2">
            {detections.length === 0 ? (
              <div className="dashboard-field text-[12px] text-[var(--text-secondary)]">No detections in the current frame.</div>
            ) : (
              detections.map((item) => {
                const active = effectiveSelectedTrackId !== null && item.key === effectiveSelectedTrackId;
                return (
                  <button
                    key={item.key}
                    type="button"
                    onClick={() => onSelectTrackId?.(active ? null : item.key)}
                    className={`dashboard-field text-left transition-colors ${
                      active ? "border-[var(--tone-coral-border)] bg-[var(--tone-coral-bg)]" : ""
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="text-[12px] font-medium text-[var(--foreground)]">{item.className}</div>
                        <div className="dashboard-micro mt-1">
                          {item.trackId || "no-track"} · conf {item.confidence?.toFixed(2) ?? "n/a"} · depth {item.depthM?.toFixed(2) ?? "n/a"}m
                        </div>
                      </div>
                      <Focus className={`size-4 ${active ? "text-[var(--tone-coral-fg)]" : "text-[var(--text-faint)]"}`} />
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>

        <div className="space-y-3">
          <div className="dashboard-panel-strong p-3.5">
            <div className="dashboard-eyebrow mb-2">Selected Object Inspector</div>
            {selectedDetection !== null || selectedTargetSummary !== null ? (
              <div className="space-y-2 text-[11px]">
                <div className="dashboard-field">
                  <div className="dashboard-eyebrow mb-1">class / track</div>
                  <div className="text-[var(--foreground)]">
                    {selectedDetection?.className || selectedTargetSummary?.className || "unknown"} /{" "}
                    {selectedDetection?.trackId || selectedTargetSummary?.trackId || "n/a"}
                  </div>
                </div>
                <div className="dashboard-field">
                  <div className="dashboard-eyebrow mb-1">confidence / depth</div>
                  <div className="text-[var(--foreground)]">
                    {selectedDetection?.confidence?.toFixed(2) ?? selectedTargetSummary?.confidence?.toFixed(2) ?? "n/a"} /{" "}
                    {selectedDetection?.depthM?.toFixed(2) ?? selectedTargetSummary?.depthM?.toFixed(2) ?? "n/a"}m
                  </div>
                </div>
                <div className="dashboard-field">
                  <div className="dashboard-eyebrow mb-1">nav goal / world pose</div>
                  <div className="dashboard-mono text-[var(--foreground)]">
                    {(selectedTargetSummary?.navGoalPixel ?? []).join(", ") || "n/a"} ·{" "}
                    {(selectedDetection?.worldPose ?? selectedTargetSummary?.worldPose ?? []).join(", ") || "n/a"}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-[12px] text-[var(--text-secondary)]">Select a detection to inspect it here.</div>
            )}
          </div>

          <div className="dashboard-panel-strong p-3.5">
            <div className="dashboard-eyebrow mb-2">Recent Frame Strip</div>
            <div className="grid grid-cols-3 gap-2">
              {filmstrip.length === 0 ? (
                <div className="col-span-3 text-[12px] text-[var(--text-secondary)]">No frame thumbnails yet.</div>
              ) : (
                filmstrip.map((item) => {
                  const active = activeFilmstripFrameId === item.frameId;
                  return (
                    <button
                      key={item.frameId}
                      type="button"
                      onClick={() => onSelectFrameId?.(active ? null : item.frameId)}
                      className={`overflow-hidden rounded-[14px] border text-left ${
                        active ? "border-[var(--tone-cyan-border)]" : "border-[rgba(var(--ink-rgb),0.08)]"
                      }`}
                    >
                      {item.imageData ? (
                        <img src={item.imageData} alt={`frame ${item.frameId}`} className="aspect-video w-full object-cover" />
                      ) : (
                        <div className="flex aspect-video items-center justify-center bg-[var(--surface-2)] text-[11px] text-[var(--text-secondary)]">
                          frame {item.frameId}
                        </div>
                      )}
                      <div className="flex items-center justify-between gap-2 px-2 py-1.5 text-[10px]">
                        <span className="dashboard-mono text-[var(--foreground)]">#{item.frameId}</span>
                        {item.hasDecision ? <ConsoleBadge tone="amber" className="!px-1.5 !py-0.5" dot={false}>S2</ConsoleBadge> : null}
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </ConsolePanel>
  );
}
