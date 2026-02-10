import { useStreaming } from '../application/streaming/StreamingProvider';
import { EMPTY_METADATA } from '../infra/streaming/metadataParsers';
import { AgentControl } from './AgentControl';
import { VideoPanel } from './realtime/VideoPanel';

export function HabitatViewer() {
  const {
    streamState,
    error,
    fps,
    hasVideo,
    metadata: rawMetadata,
    videoRef,
    attachVideoElement,
    connect,
  } = useStreaming();

  const metadata = rawMetadata ?? EMPTY_METADATA;
  const source = metadata.streamSource;
  const sourceType = source?.type ?? 'unknown';
  const sourceTopic = source?.imageTopic ?? '-';
  const sourceVideo = source?.videoPath ?? '-';

  return (
    <div className="flex flex-col space-y-6 pt-4">
      <p className="text-[14px] text-[#a0aec0]">
        Habitat 센서 스트림과 에이전트 이동 제어를 한 페이지에서 제공합니다.
      </p>

      <div className="grid min-h-0 grid-cols-[2fr_1fr] gap-6">
        <div className="panel flex min-h-[420px] flex-col">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[18px] text-[#2d3748]">Habitat Viewer</h3>
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

          <div className="min-h-0 flex-1">
            <VideoPanel
              hasVideo={hasVideo}
              videoRef={videoRef}
              overlays={[]}
              onVideoElementReady={attachVideoElement}
            />
          </div>

          {error && <p className="mt-3 text-[12px] text-red-500">{error}</p>}
        </div>

        <div className="panel">
          <h3 className="mb-3 text-[18px] text-[#2d3748]">Feed Status</h3>
          <div className="space-y-3 text-[13px] text-[#2d3748]">
            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">State</div>
              <div>{streamState}</div>
            </div>

            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Source Type</div>
              <div>{sourceType}</div>
            </div>

            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Image Topic</div>
              <div>{sourceTopic}</div>
            </div>

            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Video Source</div>
              <div className="break-all">{sourceVideo}</div>
            </div>

            <div className="rounded-lg bg-gray-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Frame</div>
              <div>#{metadata.frameIdx}</div>
            </div>
          </div>
        </div>
      </div>

      <AgentControl />
    </div>
  );
}
