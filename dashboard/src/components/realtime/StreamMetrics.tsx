import type { StreamMetadata } from '../../domain/streaming/streamTypes';

type StreamMetricsProps = {
  metadata: StreamMetadata;
};

export function StreamMetrics({ metadata }: StreamMetricsProps) {
  const metrics = [
    {
      label: 'Detected Objects',
      value: (metadata.entities ?? []).length.toString(),
      context: 'current frame'
    },
    {
      label: 'Entity Records',
      value: (metadata.entityRecords ?? []).length.toString(),
      context: 'tracking'
    }
  ];
  const focusTargets = Array.isArray(metadata.focusTargets) ? metadata.focusTargets : [];

  return (
    <div className="mt-6 grid grid-cols-3 gap-4">
      {metrics.map((metric) => (
        <div key={metric.label} className="rounded-lg bg-gray-50 p-4">
          <div className="text-[24px] text-[#2d3748]">{metric.value}</div>
          <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">
            {metric.label}
          </div>
          <div className="text-[10px] text-[#a0aec0]">{metric.context}</div>
        </div>
      ))}

      {/* Focus Targets 카드 */}
      <div className="rounded-lg bg-gray-50 p-4">
        {focusTargets.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
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
  );

}
