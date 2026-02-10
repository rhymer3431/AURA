import { useEffect, useMemo, useState, type MutableRefObject } from 'react';

import type { StreamMetadata, StreamState } from '../../domain/streaming/streamTypes';
import {
  buildBoundingBoxOverlays,
  type FrameSize
} from '../../domain/visualization/boundingBoxes';
import { EMPTY_METADATA } from '../../infra/streaming/metadataParsers';
import { VideoPanel } from './VideoPanel';

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
    () => buildBoundingBoxOverlays(safeMetadata.entities ?? [], frameSize ?? undefined),
    [safeMetadata.entities, frameSize],
  );

  return (
    <div className="panel">

      {/* Header */}
      <div className="mb-4 flex items-center justify-between shrink-0">
        <h3 className="text-[18px] text-[#2d3748]">Streaming</h3>

        <div className="flex items-center gap-2">
          <div className="rounded-lg bg-[#4fd1c5] px-3 py-1">
            <span className="text-[12px] text-white">{fps.toFixed(1)} FPS</span>
          </div>

          <button
            onClick={onReconnect}
            className="rounded-lg border border-[#e2e8f0] bg-white px-3 py-1 text-[12px] text-[#2d3748] transition-colors hover:bg-gray-50"
          >
            Connect
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

      {/* Caption Only */}
      <div className="mt-6 rounded-lg bg-gray-50 px-6 py-6">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-[#4fd1c5]" />
            <span className="text-[12px] uppercase tracking-wide text-[#a0aec0]">
              Caption
            </span>
          </div>
        </div>

        {captionText ? (
          <p className="mt-1 whitespace-pre-line text-[17px] leading-relaxed text-[#2d3748] font-medium">
            {captionText}
          </p>
        ) : (
          <p className="mt-1 text-[14px] italic text-[#a0aec0]">
            Waiting for caption from stream...
          </p>
        )}
      </div>
    </div>
  );
}