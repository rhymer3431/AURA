import { Fragment } from 'react';

export type RealtimeEvent = {
  id: string;
  time: string;
  source: {
    code: string;
    color: string;
  };
  level: 'info' | 'warning';
  title: string;
  details: {
    text: string;
    accentColor?: string;
  }[];
  isWarning?: boolean;
};

type EventItemProps = {
  event: RealtimeEvent;
};

export function EventItem({ event }: EventItemProps) {
  const containerClasses = event.isWarning
    ? 'rounded-lg border border-[#f6ad55]/40 bg-[#f6ad55]/5 p-3'
    : 'rounded-lg border border-[#e2e8f0] p-3';

  return (
    <div className={containerClasses}>
      <div className="mb-2 flex items-center justify-between">
        <div className="text-[12px] text-[#a0aec0]">{event.time}</div>
        <div className="flex items-center gap-2">
          <span
            className="rounded-md px-2 py-0.5 text-[10px]"
            style={{
              backgroundColor: `${event.source.color}1A`,
              color: event.source.color
            }}
          >
            {event.source.code}
          </span>
          <span
            className={`rounded-md px-2 py-0.5 text-[10px] text-white ${
              event.level === 'warning' ? 'bg-[#f6ad55]' : 'bg-[#4fd1c5]'
            }`}
          >
            {event.level}
          </span>
        </div>
      </div>

      <div className="mb-1 text-[13px] text-[#2d3748]">{event.title}</div>
      <div className="flex items-center gap-2 text-[12px] text-[#a0aec0]">
        {event.details.map((detail, index) => (
          <Fragment key={`${event.id}-${detail.text}-${index}`}>
            {index > 0 && <span className="text-[#e2e8f0]">|</span>}
            <span style={detail.accentColor ? { color: detail.accentColor } : undefined}>
              {detail.text}
            </span>
          </Fragment>
        ))}
      </div>
    </div>
  );
}
