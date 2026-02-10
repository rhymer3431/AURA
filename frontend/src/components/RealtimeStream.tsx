import { useMemo } from 'react';
import { useStreaming } from '../application/streaming/StreamingProvider';
import { EMPTY_METADATA } from '../domain/streaming/metadataParsers';
import type { StreamEntity, StreamEntityRecord } from '../domain/streaming/streamTypes';
import type { RealtimeEvent } from './realtime/EventItem';
import { EventsPanel } from './realtime/EventsPanel';
import { LiveStreamPanel } from './realtime/LiveStreamPanel';

export function RealtimeStream() {
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

  const isStreamEntity = (record: StreamEntity | StreamEntityRecord): record is StreamEntity =>
    'trackId' in record && 'cls' in record;

  const isStreamEntityRecord = (record: StreamEntity | StreamEntityRecord): record is StreamEntityRecord =>
    'trackHistory' in record && 'baseCls' in record;

  const realtimeEvents: RealtimeEvent[] = useMemo(() => {
    if (metadata.entityRecords.length === 0 && metadata.entities.length === 0) {
      return [
        {
          id: 'placeholder-event',
          time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          source: { code: 'SYS', color: '#4fd1c5' },
          level: streamState === 'connecting' ? 'warning' : 'info',
          title: streamState === 'connecting' ? 'Connecting to stream' : 'Awaiting metadata',
          details: [{ text: 'No metadata yet' }]
        }
      ];
    }

    return (metadata.entityRecords.length ? metadata.entityRecords : metadata.entities)
      .slice(-5)
      .map((record, idx) => {
        const now = new Date();
        const code = isStreamEntityRecord(record) ? record.baseCls : record.cls;
        const entityId = record.entityId ?? idx;
        const trackText = isStreamEntity(record)
          ? record.trackId
          : record.trackHistory && record.trackHistory.length > 0
            ? record.trackHistory[record.trackHistory.length - 1]
            : idx;
        return {
          id: `event-${entityId}-${idx}`,
          time: now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          source: { code: (code || 'EVT').slice(0, 3).toUpperCase(), color: '#4fd1c5' },
          level: 'info',
          title: 'Track Update',
          details: [
            { text: code },
            { text: `Track #${trackText}` },
            { text: `Frame ${metadata.frameIdx}` }
          ]
        };
      });
  }, [metadata, streamState]);

  return (
    <div className="flex h-full flex-col space-y-6">
      <p className="text-[14px] text-[#a0aec0]">
        Live video and perception events from the robot
      </p>

      <div className="grid h-full min-h-0 grid-cols-[2fr_1fr] gap-6">
        <LiveStreamPanel
          streamState={streamState}
          error={error}
          fps={fps}
          hasVideo={hasVideo}
          metadata={metadata}
          videoRef={videoRef}
          onVideoElementReady={attachVideoElement}
          onReconnect={connect}
        />
        <EventsPanel events={realtimeEvents} />
      </div>
    </div>
  );
}
