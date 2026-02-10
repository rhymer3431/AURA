import type { RefObject } from 'react';
import { Play, RefreshCw, Volume2 } from 'lucide-react';
import type { StreamMetadata, StreamState } from '../../domain/streaming/streamTypes';

type VideoPlaceholderProps = {
  streamState: StreamState;
  metadata: StreamMetadata;
  videoRef: RefObject<HTMLVideoElement | null>;
  hasVideo: boolean;
};

export function VideoPlaceholder({ streamState, metadata, videoRef, hasVideo }: VideoPlaceholderProps) {
  const activeEntities = metadata.entities.slice(0, 4);
  const showOverlay = activeEntities.length > 0;

  return (
    <div className="relative w-full overflow-hidden rounded-lg bg-gradient-to-br from-gray-800 to-gray-900 aspect-video">
      <video ref={videoRef} className="absolute inset-0 h-full w-full object-cover" playsInline muted />

      {!hasVideo && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div
            className="absolute inset-0 opacity-10"
            style={{
              backgroundImage:
                'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
              backgroundSize: '40px 40px'
            }}
          ></div>
          <div className="relative text-center text-gray-400">
            <div className="mb-2 text-[14px]">Video Stream (WebRTC)</div>
            <div className="text-[12px] text-gray-500">
              {streamState === 'connecting'
                ? 'Connecting...'
                : streamState === 'error'
                ? 'Stream error'
                : 'Awaiting first frame'}
            </div>
          </div>
        </div>
      )}

      {showOverlay && (
        <div className="absolute left-4 top-4 flex flex-wrap gap-2">
          {activeEntities.map((entity) => (
            <div
              key={`${entity.cls}-${entity.entityId}`}
              className="rounded-md px-2 py-1 text-[11px] text-white backdrop-blur-sm"
              style={{ backgroundColor: '#4fd1c5e6' }}
            >
              {entity.cls} #{entity.entityId}
            </div>
          ))}
        </div>
      )}

      <div className="absolute bottom-4 right-4 flex items-center gap-2">
        <button className="rounded-md bg-black/60 p-2 text-white transition-colors backdrop-blur-sm hover:bg-black/80">
          <Play className="size-4 fill-white" />
        </button>
        <button className="rounded-md bg-black/60 p-2 text-white transition-colors backdrop-blur-sm hover:bg-black/80">
          <RefreshCw className="size-4" />
        </button>
        <button className="rounded-md bg-black/60 p-2 text-white transition-colors backdrop-blur-sm hover:bg-black/80">
          <Volume2 className="size-4" />
        </button>
      </div>
    </div>
  );
}
