import { ChevronDown, Search } from 'lucide-react';
import type { MemoryRecordView } from './types';

type StateFilter = MemoryRecordView['state'] | 'All';

type MemoryRecordsPanelProps = {
  records: MemoryRecordView[];
  searchTerm: string;
  onSearchTermChange: (value: string) => void;
  stateFilter: StateFilter;
  onStateFilterChange: (value: StateFilter) => void;
  selectedEntityId: number | null;
  onSelectEntity: (entityId: number) => void;
  entityCount: number;
  hasRecords: boolean;
};

const FILTER_OPTIONS: StateFilter[] = ['All', 'Active', 'Dormant', 'Pruned'];

export function MemoryRecordsPanel({
  records,
  searchTerm,
  onSearchTermChange,
  stateFilter,
  onStateFilterChange,
  selectedEntityId,
  onSelectEntity,
  entityCount,
  hasRecords,
}: MemoryRecordsPanelProps) {
  return (
    <div className="panel">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-[18px] text-[#2d3748]">Entity Long-Term Memory</h3>
        <div className="rounded-lg bg-gray-50 px-3 py-1 text-[11px] uppercase tracking-wide text-[#a0aec0]">
          {entityCount} records
        </div>
      </div>

      <div className="mb-4 flex flex-row items-center gap-3">
        <div className="flex flex-1 items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
          <Search className="size-4 text-[#a0aec0]" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => onSearchTermChange(e.target.value)}
            placeholder="Search by ID, label, or state"
            className="flex-1 border-0 bg-transparent text-[14px] text-[#2d3748] outline-none"
          />
        </div>
        <button
          className="flex w-auto items-center justify-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-4 py-2 text-[14px] text-[#2d3748] transition-colors hover:bg-gray-50"
        >
          <span>Last seen</span>
          <ChevronDown className="size-4" />
        </button>
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        {FILTER_OPTIONS.map((option) => (
          <button
            key={option}
            onClick={() => onStateFilterChange(option)}
            className={`rounded-lg px-3 py-1 text-[12px] transition-colors ${stateFilter === option
              ? 'bg-[#4fd1c5] text-white'
              : 'bg-gray-100 text-[#a0aec0] hover:bg-gray-200'
              }`}
          >
            {option}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto">
        <div className="min-w-[820px]">
          <div className="mb-3 grid grid-cols-[80px_1fr_90px_120px_150px_100px] gap-3 border-b border-[#e2e8f0] pb-3">
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">ID</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Class</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Last Seen</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Seen Count</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Stability</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">State</div>
          </div>

          <div className="space-y-2">
            {records.length === 0 ? (
              <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-[#e2e8f0] bg-gray-50/60 p-6 text-center text-[13px] text-[#a0aec0]">
                {hasRecords
                  ? 'No records match the current filters.'
                  : 'No entity records received yet. Waiting for streaming metadata...'}
              </div>
            ) : (
              records.map((entity) => (
                <div
                  key={entity.entityId}
                  onClick={() => onSelectEntity(entity.entityId)}
                  className={`grid grid-cols-[80px_1fr_90px_120px_150px_100px] cursor-pointer items-center gap-3 rounded-lg px-3 py-3 transition-colors ${selectedEntityId === entity.entityId
                    ? 'border border-[#4fd1c5] bg-[#4fd1c5]/10'
                    : 'border border-transparent hover:bg-gray-50'
                    }`}
                >
                  <div className="text-[14px] text-[#2d3748]">#{entity.entityId}</div>
                  <div className="flex items-center gap-2">
                    <div
                      className="flex size-8 items-center justify-center rounded-full text-center text-[11px] text-white"
                      style={{ backgroundColor: entity.avatar }}
                    >
                      {entity.cls[0]?.toUpperCase?.() ?? '?'}
                    </div>
                    <div className="text-[14px] text-[#2d3748] capitalize">{entity.cls}</div>
                  </div>
                  <div className="text-[12px] text-[#a0aec0]">{entity.lastSeenFrame}</div>
                  <div className="text-[12px] text-[#a0aec0]">{entity.seenCount}</div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1">
                      <div className="h-1 w-full rounded-full bg-[#e2e8f0]">
                        <div
                          className="h-1 rounded-full bg-[#4fd1c5]"
                          style={{ width: `${entity.stabilityScore * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    <div className="w-8 text-[11px] text-[#4fd1c5]">
                      {Math.round(entity.stabilityScore * 100)}
                    </div>
                  </div>
                  <div>
                    <span
                      className={`inline-block rounded-md px-2 py-1 text-[11px] text-white ${entity.state === 'Active'
                        ? 'bg-[#4fd1c5]'
                        : entity.state === 'Dormant'
                          ? 'bg-[#4299e1]'
                          : 'bg-[#fc8181]'
                        }`}
                    >
                      {entity.state}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
