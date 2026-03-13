import { useEffect, useRef, useState } from "react";
import { Maximize2, Eye, EyeOff, Video, SignalHigh } from "lucide-react";

import { useDashboard } from "../state";
import { useWebRTCViewer } from "../hooks/useWebRTCViewer";
import { asArray, asRecord, formatMs, numberValue, stringValue } from "../selectors";
import { buildApiUrl } from "../network";

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
  const showDepth = state?.session.config?.showDepth === true && trackRoles.includes("depth");
  const detections = asArray<Record<string, unknown>>(telemetry.detections);
  const waitingForFrame = stringValue(snapshot.type) === "waiting_for_frame";
  const frameSource = stringValue(snapshot.source, stringValue(state?.sensors.source, "aura_runtime"));
  const detectorBackend = stringValue(snapshot.detector_backend, stringValue(state?.perception.detectorBackend, "unknown"));
  const peerSessionId = stringValue(state?.transport.peerSessionId, "none");

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Video className="size-4 text-black/40" />
          <h3 className="text-[15px] font-semibold text-black">Live Robot View</h3>
          <span
            className={`ml-2 inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-medium border ${
              viewer.connected
                ? "bg-emerald-50 text-emerald-700 border-emerald-100"
                : "bg-amber-50 text-amber-700 border-amber-100"
            }`}
          >
            <span className={`size-1.5 rounded-full ${viewer.connected ? "bg-emerald-500" : "bg-amber-500"}`} />
            {viewer.connected ? "WEBRTC" : viewerEnabled ? "CONNECTING" : "INACTIVE"}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOverlay((current) => !current)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors border ${
              showOverlay
                ? "bg-black/5 border-black/10 text-black/80"
                : "bg-white border-black/10 text-black/40 hover:bg-black/[0.02]"
            }`}
          >
            {showOverlay ? <Eye className="size-3.5" /> : <EyeOff className="size-3.5" />}
            BBox Overlays
          </button>
          <button className="p-1.5 rounded-lg border border-black/10 text-black/40 hover:bg-black/[0.02] transition-colors">
            <Maximize2 className="size-3.5" />
          </button>
        </div>
      </div>

      <div className="relative w-full aspect-video bg-neutral-950 rounded-lg overflow-hidden border border-black/10">
        <video
          ref={viewer.rgbVideoRef}
          className="w-full h-full object-cover"
          autoPlay
          muted
          playsInline
        />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />

        {showDepth && (
          <div className="absolute bottom-3 right-3 w-[28%] aspect-video rounded-lg overflow-hidden border border-white/15 bg-black/40 backdrop-blur-sm">
            <video
              ref={viewer.depthVideoRef}
              className="w-full h-full object-cover"
              autoPlay
              muted
              playsInline
            />
          </div>
        )}

        <div className="absolute top-0 left-0 w-full p-3 flex justify-between items-start pointer-events-none">
          <div className="flex flex-col gap-1">
            <div className="font-mono text-[10px] text-white/80 bg-black/40 px-2 py-0.5 rounded backdrop-blur-sm">
              SRC: {frameSource}
            </div>
            <div className="font-mono text-[10px] text-white/80 bg-black/40 px-2 py-0.5 rounded backdrop-blur-sm">
              RES: {numberValue(image.width) ?? 0}x{numberValue(image.height) ?? 0} | DEPTH: {showDepth ? "on" : "off"}
            </div>
          </div>
          <div className="flex items-center gap-1.5 font-mono text-[10px] text-white/80 bg-black/40 px-2 py-0.5 rounded backdrop-blur-sm">
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

      <div className="mt-3 flex items-center justify-between px-1 gap-4">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="text-[11px] text-black/60">
            Inference: <span className="text-black/80 font-medium">{detectorBackend}</span>
          </div>
          <div className="w-px h-3 bg-black/10" />
          <div className="text-[11px] text-black/40">
            Detected: <span className="text-black/80 font-medium">{detections.length} objects</span>
          </div>
          <div className="w-px h-3 bg-black/10" />
          <div className="text-[11px] text-black/40">
            Trajectory: <span className="text-black/80 font-medium">{asArray(telemetry.trajectoryPixels ?? telemetry.trajectory_pixels).length} pts</span>
          </div>
        </div>
        <div className="text-[10px] text-black/30 font-mono">
          peer {peerSessionId}
        </div>
      </div>
    </div>
  );
}
