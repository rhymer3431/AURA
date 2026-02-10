import { useEffect, useMemo, useState } from 'react';
import { useStreaming } from '../application/streaming/StreamingProvider';
import { EMPTY_METADATA } from '../infra/streaming/metadataParsers';
import { VideoPanel } from './realtime/VideoPanel';

type PosePoint = {
  x: number;
  y: number;
  z: number;
};

const MAX_POSE_POINTS = 400;
const MIN_POINT_DELTA = 0.005;

export function SlamViewer() {
  const {
    streamState,
    fps,
    hasVideo,
    metadata: rawMetadata,
    videoRef,
    attachVideoElement,
    connect,
  } = useStreaming();

  const metadata = rawMetadata ?? EMPTY_METADATA;
  const slamPose = metadata.slamPose;
  const [history, setHistory] = useState<PosePoint[]>([]);

  useEffect(() => {
    if (!slamPose) return;

    const next: PosePoint = {
      x: slamPose.position.x,
      y: slamPose.position.y,
      z: slamPose.position.z,
    };

    setHistory((prev) => {
      const last = prev[prev.length - 1];
      if (last) {
        const dx = next.x - last.x;
        const dy = next.y - last.y;
        const dz = next.z - last.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < MIN_POINT_DELTA) {
          return prev;
        }
      }

      const merged = [...prev, next];
      if (merged.length > MAX_POSE_POINTS) {
        return merged.slice(merged.length - MAX_POSE_POINTS);
      }
      return merged;
    });
  }, [slamPose]);

  const trajectory = useMemo(() => {
    const width = 1000;
    const height = 560;
    const padding = 28;

    if (history.length === 0) {
      return {
        width,
        height,
        points: '',
        current: null as { x: number; y: number } | null,
        start: null as { x: number; y: number } | null,
      };
    }

    const xs = history.map((p) => p.x);
    const zs = history.map((p) => p.z);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    const spanX = Math.max(maxX - minX, 0.01);
    const spanZ = Math.max(maxZ - minZ, 0.01);
    const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanZ);

    const mapPoint = (p: PosePoint) => ({
      x: padding + (p.x - minX) * scale,
      y: height - padding - (p.z - minZ) * scale,
    });

    const mapped = history.map(mapPoint);
    const points = mapped.map((p) => `${p.x},${p.y}`).join(' ');
    const current = mapped[mapped.length - 1];
    const start = mapped[0];

    return {
      width,
      height,
      points,
      current,
      start,
    };
  }, [history]);

  return (
    <div className="flex h-full flex-col space-y-6 pt-4">
      <p className="text-[14px] text-[#a0aec0]">
        Live ORB-SLAM trajectory and pose from {metadata.slamPoseTopic ?? '/orbslam/pose'}
      </p>

      <div className="grid h-full min-h-0 grid-cols-[1.35fr_1fr] gap-6">
        <div className="panel flex min-h-0 flex-col">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[18px] text-[#2d3748]">SLAM Trajectory</h3>
            <div className="flex items-center gap-2">
              <div className="rounded-lg bg-[#4fd1c5] px-3 py-1">
                <span className="text-[12px] text-white">{fps.toFixed(1)} FPS</span>
              </div>
              <button
                onClick={connect}
                className="rounded-lg border border-[#e2e8f0] bg-white px-3 py-1 text-[12px] text-[#2d3748] transition-colors hover:bg-gray-50"
              >
                Connect
              </button>
            </div>
          </div>

          <div className="relative flex-1 overflow-hidden rounded-xl border border-[#e2e8f0] bg-gradient-to-br from-gray-50 to-white">
            <svg viewBox={`0 0 ${trajectory.width} ${trajectory.height}`} className="h-full w-full">
              <defs>
                <pattern id="slam-grid" width="50" height="50" patternUnits="userSpaceOnUse">
                  <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#e2e8f0" strokeWidth="1" />
                </pattern>
              </defs>
              <rect width={trajectory.width} height={trajectory.height} fill="url(#slam-grid)" />
              {trajectory.points && (
                <polyline
                  fill="none"
                  stroke="#4fd1c5"
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={trajectory.points}
                />
              )}
              {trajectory.start && (
                <circle cx={trajectory.start.x} cy={trajectory.start.y} r="8" fill="#4299e1" />
              )}
              {trajectory.current && (
                <circle cx={trajectory.current.x} cy={trajectory.current.y} r="9" fill="#2d3748" />
              )}
            </svg>

            {history.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center text-[14px] text-[#a0aec0]">
                Waiting for SLAM pose...
              </div>
            )}
          </div>

          <div className="mt-3 text-[11px] text-[#a0aec0]">
            state: {streamState} | trajectory points: {history.length}
          </div>
        </div>

        <div className="flex min-h-0 flex-col gap-6">
          <div className="panel">
            <h3 className="mb-3 text-[18px] text-[#2d3748]">Current Pose</h3>
            {slamPose ? (
              <div className="space-y-3 text-[13px] text-[#2d3748]">
                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Position</div>
                  <div>x: {slamPose.position.x.toFixed(4)}</div>
                  <div>y: {slamPose.position.y.toFixed(4)}</div>
                  <div>z: {slamPose.position.z.toFixed(4)}</div>
                </div>
                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Quaternion</div>
                  <div>x: {slamPose.orientation.x.toFixed(4)}</div>
                  <div>y: {slamPose.orientation.y.toFixed(4)}</div>
                  <div>z: {slamPose.orientation.z.toFixed(4)}</div>
                  <div>w: {slamPose.orientation.w.toFixed(4)}</div>
                </div>
              </div>
            ) : (
              <p className="text-[14px] text-[#a0aec0]">No pose received yet.</p>
            )}
          </div>

          <div className="panel flex min-h-0 flex-1 flex-col">
            <h3 className="mb-3 text-[18px] text-[#2d3748]">SLAM Camera Preview</h3>
            <div className="min-h-0 flex-1">
              <VideoPanel
                hasVideo={hasVideo}
                videoRef={videoRef}
                overlays={[]}
                onVideoElementReady={attachVideoElement}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
