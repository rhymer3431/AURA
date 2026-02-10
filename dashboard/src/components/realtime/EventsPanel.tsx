import type { StreamMetadata } from '../../domain/streaming/streamTypes';
import { EMPTY_METADATA } from '../../infra/streaming/metadataParsers';
import { EventItem, type RealtimeEvent } from './EventItem';

type EventsPanelProps = {
  events: RealtimeEvent[];
  metadata: StreamMetadata | null | undefined;
};

export function EventsPanel({ events, metadata }: EventsPanelProps) {
  const safeMetadata = metadata ?? EMPTY_METADATA;
  const slamPose = safeMetadata.slamPose;
  const slamPositionText = slamPose
    ? `x:${slamPose.position.x.toFixed(2)} y:${slamPose.position.y.toFixed(2)} z:${slamPose.position.z.toFixed(2)}`
    : 'Waiting';
  const sourceType = safeMetadata.streamSource?.type ?? 'unknown';
  const habitatFeedText = sourceType === 'ros2' ? 'Live ROS2 feed' : sourceType.toUpperCase();
  const habitatFeedContext =
    safeMetadata.streamSource?.imageTopic
      ? `topic ${safeMetadata.streamSource.imageTopic}`
      : (safeMetadata.streamSource?.videoPath ?? 'no source metadata');

  const metrics = [
    {
      label: 'Detected Objects',
      value: (safeMetadata.entities ?? []).length.toString(),
      context: 'current frame'
    },
    {
      label: 'Entity Records',
      value: (safeMetadata.entityRecords ?? []).length.toString(),
      context: 'tracking'
    },
    {
      label: 'Habitat Feed',
      value: habitatFeedText,
      context: habitatFeedContext
    },
    {
      label: 'SLAM Pose',
      value: slamPositionText,
      context: safeMetadata.slamPoseTopic ?? '/orbslam/pose'
    }
  ];

  const focusTargets = Array.isArray(safeMetadata.focusTargets) ? safeMetadata.focusTargets : [];

  return (
    <div className="flex min-h-0 flex-col">
      <div className="panel">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-[18px] text-[#2d3748]">Realtime Events</h3>
        </div>


        <div className="flex-1 space-y-3 overflow-y-auto mb-6">
          {events.map((event) => (
            <EventItem key={event.id} event={event} />
          ))}
        </div>

        {/* Metrics Section - StreamMetrics Style */}
        <div className="border-t border-[#e2e8f0] pt-6 mt-6">
          <div className="grid grid-cols-2 gap-4">
            {metrics.map((metric) => (
              <div key={metric.label} className="rounded-lg bg-gray-50 p-4">
                <div className="text-[16px] text-[#2d3748] break-words">{metric.value}</div>
                <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">
                  {metric.label}
                </div>
                <div className="text-[10px] text-[#a0aec0]">{metric.context}</div>
              </div>
            ))}
          </div>

          <div className="mt-4 rounded-lg bg-gray-50 p-4">
            {focusTargets.length > 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                {focusTargets.map((target, idx) => (
                  <span
                    key={`${target}-${idx}`}
                    className="rounded-full bg-white px-3 py-1 text-[12px] text-[#2d3748] shadow-sm"
                  >
                    {target}
                  </span>
                ))}
              </div>
            )}
            <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">
              Focus Targets
            </div>
            <div className="text-[10px] text-[#a0aec0]">
              {focusTargets.length === 0
                ? "No focus targets detected."
                : "Active focus targets in current frame."}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
