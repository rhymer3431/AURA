import { useEffect, useMemo, useState, type MutableRefObject } from 'react';
import { EMPTY_METADATA } from '../../domain/streaming/metadataParsers';
import type { StreamMetadata, StreamState } from '../../domain/streaming/streamTypes';
import {
  buildBoundingBoxOverlays,
  type BoundingBoxOverlay,
  type FrameSize,
} from '../../domain/visualization/boundingBoxes';
import { StreamMetrics } from './StreamMetrics';

type LiveStreamPanelProps = {
  streamState: StreamState;
  fps: number;
  hasVideo: boolean;
  metadata: StreamMetadata | null | undefined;
  videoRef: MutableRefObject<HTMLVideoElement | null>;
  error?: string | null;
  onReconnect: () => void;
  onVideoElementReady?: (el: HTMLVideoElement | null) => void;
};

export function LiveStreamPanel({
  streamState,
  fps,
  hasVideo,
  metadata,
  videoRef,
  error,
  onReconnect,
  onVideoElementReady,
}: LiveStreamPanelProps) {

  const safeMetadata = metadata ?? EMPTY_METADATA;
  const [frameSize, setFrameSize] = useState<FrameSize | null>(null);
  const isLive = streamState === 'playing';
  const captionText = (safeMetadata.caption ?? '').trim();

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateSize = () => {
      const width = video.videoWidth || video.clientWidth;
      const height = video.videoHeight || video.clientHeight;
      if (width && height) {
        setFrameSize((prev) =>
          prev && prev.width === width && prev.height === height ? prev : { width, height },
        );
      }
    };

    updateSize();
    video.addEventListener('loadedmetadata', updateSize);

    const resizeObserver = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(updateSize) : null;
    resizeObserver?.observe(video);

    return () => {
      video.removeEventListener('loadedmetadata', updateSize);
      resizeObserver?.disconnect();
    };
  }, [videoRef, hasVideo]);

  const overlays = useMemo(
    () => buildBoundingBoxOverlays(safeMetadata.entities, frameSize ?? undefined),
    [safeMetadata.entities, frameSize],
  );

  return (
    <div className="flex h-full min-h-0 flex-col rounded-[15px] bg-white p-6 shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">

      {/* Header */}
      <div className="mb-4 flex items-center justify-between shrink-0">
        <h3 className="text-[18px] text-[#2d3748]">Realtime Perception Stream</h3>

        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-2 rounded-lg px-3 py-1 ${isLive ? 'bg-red-600' : 'bg-gray-200'}`}>
            <div className={`size-2 rounded-full ${isLive ? 'bg-white animate-pulse' : 'bg-gray-400'}`} />
            <span className={`text-[12px] ${isLive ? 'text-white' : 'text-gray-600'}`}>
              {isLive ? 'LIVE' : streamState.toUpperCase()}
            </span>
          </div>

          <div className="rounded-lg bg-[#4fd1c5] px-3 py-1">
            <span className="text-[12px] text-white">{fps.toFixed(1)} FPS</span>
          </div>

          <button
            onClick={onReconnect}
            className="rounded-lg border border-[#e2e8f0] bg-white px-3 py-1 text-[12px] text-[#2d3748] transition-colors hover:bg-gray-50"
          >
            Reconnect
          </button>
        </div>
      </div>

      {/* Video full height */}
      <div className="flex-1 min-h-0">
        <VideoPanel
          hasVideo={hasVideo}
          videoRef={videoRef}
          overlays={overlays}
          onVideoElementReady={onVideoElementReady}
        />
      </div>

      {/* Caption Expanded */}
      <div className="mt-4 rounded-lg border border-[#e2e8f0] bg-white p-5 shadow-inner">
        <div className="mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-[#4fd1c5]" />
            <span className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Realtime Caption</span>
          </div>
          <span className="text-[11px] text-[#a0aec0]">Frame {safeMetadata.frameIdx}</span>
        </div>
        <p className="text-[16px] leading-relaxed text-[#2d3748]">
          {captionText || 'Waiting for caption from stream...'}
        </p>
      </div>

      {/* Metrics at bottom */}
      <div className="shrink-0 mt-4">
        <StreamMetrics metadata={safeMetadata} />
      </div>
    </div>
  );
}


/* ===============================
      Video Panel Component
================================== */

type VideoPanelProps = {
  hasVideo: boolean;
  videoRef: MutableRefObject<HTMLVideoElement | null>;
  overlays: BoundingBoxOverlay[];
  onVideoElementReady?: (el: HTMLVideoElement | null) => void;
};
function VideoPanel({ hasVideo, videoRef, overlays, onVideoElementReady }: VideoPanelProps) {
  const handleVideoRef = (el: HTMLVideoElement | null) => {
    videoRef.current = el;
    onVideoElementReady?.(el);
  };

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border border-white/40 bg-white/20 backdrop-blur-lg shadow-[0_8px_20px_rgba(0,0,0,0.12)]">

      {/* Video */}
      <video
        ref={handleVideoRef}
        className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-500 ${hasVideo ? "opacity-100" : "opacity-0"
          }`}
        autoPlay
        playsInline
        muted
      />

      {/* Placeholder */}
      {!hasVideo && (
        <div className="absolute inset-0 bg-gradient-to-br from-gray-200/50 to-gray-300/30 animate-pulse flex items-center justify-center">
          <p className="text-gray-600 text-sm tracking-wide">
            Connecting Video...
          </p>
        </div>
      )}

      {/* Entity overlays */}
      {overlays.length > 0 && (
        <div className="pointer-events-none absolute inset-0 z-10">
          {overlays.map((overlay) => (
            <div
              key={overlay.id}
              className="absolute rounded-md border-2"
              style={{
                left: `${overlay.box.leftPct * 100}%`,
                top: `${overlay.box.topPct * 100}%`,
                width: `${overlay.box.widthPct * 100}%`,
                height: `${overlay.box.heightPct * 100}%`,
                borderColor: overlay.color,
                boxShadow: `0 0 0 1px ${overlay.color}40`,
              }}
            >
              <div
                className="absolute left-0 top-0 flex items-center gap-2 rounded-md px-2 py-1 text-[11px] font-semibold"
                style={{ backgroundColor: `${overlay.color}e6`, color: '#0b1221' }}
              >
                <span>{overlay.label}</span>
                {typeof overlay.confidence === 'number' && !Number.isNaN(overlay.confidence) && (
                  <span className="text-[10px] opacity-80">
                    {(overlay.confidence > 1
                      ? overlay.confidence
                      : overlay.confidence * 100
                    ).toFixed(1)}
                    {overlay.confidence <= 1 ? '%' : ''}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Overlay frame */}
      <div className="pointer-events-none absolute inset-0 z-0 bg-gradient-to-b from-white/10 via-transparent to-white/20" />
    </div>
  );
}
