import { EventItem, type RealtimeEvent } from './EventItem';

type EventsPanelProps = {
  events: RealtimeEvent[];
};

export function EventsPanel({ events }: EventsPanelProps) {
  return (
    <div className="flex min-h-0 flex-col">
      <div className="flex h-full flex-col rounded-[15px] bg-white p-6 shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-[18px] text-[#2d3748]">Realtime Events</h3>
        </div>

        <div className="mb-4 flex flex-wrap gap-2">
          <button className="rounded-lg bg-[#4fd1c5] px-3 py-1 text-[12px] text-white">
            All
          </button>
          <button className="rounded-lg bg-gray-100 px-3 py-1 text-[12px] text-[#a0aec0] hover:bg-gray-200">
            Detections
          </button>
          <button className="rounded-lg bg-gray-100 px-3 py-1 text-[12px] text-[#a0aec0] hover:bg-gray-200">
            Tracking
          </button>
          <button className="rounded-lg bg-gray-100 px-3 py-1 text-[12px] text-[#a0aec0] hover:bg-gray-200">
            LLM
          </button>
          <button className="rounded-lg bg-gray-100 px-3 py-1 text-[12px] text-[#a0aec0] hover:bg-gray-200">
            Warnings
          </button>
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto">
          {events.map((event) => (
            <EventItem key={event.id} event={event} />
          ))}
        </div>

        <div className="mt-4 flex items-center justify-between border-t border-[#e2e8f0] pt-3">
          <label className="flex items-center gap-2 text-[12px] text-[#a0aec0]">
            <input type="checkbox" className="rounded border-[#e2e8f0]" defaultChecked />
            Auto-scroll
          </label>
          <div className="text-[12px] text-[#a0aec0]">Showing latest 100 events</div>
        </div>
      </div>
    </div>
  );
}
