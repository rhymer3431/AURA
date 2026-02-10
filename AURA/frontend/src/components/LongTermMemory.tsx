import { useEffect, useMemo, useState } from 'react';
import { Search, ChevronDown, FileText } from 'lucide-react';
import { useStreaming } from '../application/streaming/StreamingProvider';
import { EMPTY_METADATA } from '../domain/streaming/metadataParsers';
import type { StreamEntityRecord } from '../domain/streaming/streamTypes';

type MemoryRecordView = {
  entityId: number;
  trackId: number;
  cls: string;
  firstSeenShort: string;
  firstSeenLong: string;
  lastSeenShort: string;
  lastSeenLong: string;
  totalObservations: number;
  stabilityScore: number;
  lastPosition: string;
  state: 'Active' | 'Dormant' | 'Pruned';
  avatar: string;
};

const COLOR_PALETTE = [
  '#4fd1c5',
  '#22c55e',
  '#3b82f6',
  '#f59e0b',
  '#ef4444',
  '#a855f7',
  '#06b6d4',
  '#ec4899',
];

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

const pickColor = (entityId: number | undefined, trackId: number | undefined, idx: number) => {
  const key = trackId ?? entityId ?? idx;
  const paletteIdx = Math.abs(key) % COLOR_PALETTE.length;
  return COLOR_PALETTE[paletteIdx];
};

const readMetaField = <T,>(meta: StreamEntityRecord['meta'], keys: string[]): T | undefined => {
  if (!meta) return undefined;
  for (const key of keys) {
    const value = meta[key];
    if (value !== undefined) {
      return value as T;
    }
  }
  return undefined;
};

const buildTimeLabels = (value: unknown, fallbackFrame?: number) => {
  if (typeof value === 'string' || typeof value === 'number') {
    const asDate = new Date(value);
    if (!Number.isNaN(asDate.getTime())) {
      return {
        short: asDate.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        long: asDate.toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        }),
      };
    }
  }
  const label =
    typeof fallbackFrame === 'number' ? `Frame ${fallbackFrame.toLocaleString()}` : 'N/A';
  return { short: label, long: label };
};

const describeBoxPosition = (box?: number[]) => {
  if (!Array.isArray(box) || box.length < 4) return 'unknown position';
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const horizontal = cx < 0.33 ? 'left side' : cx < 0.66 ? 'center' : 'right side';
  const vertical = cy < 0.33 ? 'top' : cy < 0.66 ? 'center' : 'bottom';
  return `${vertical} ${horizontal}`;
};

const deriveState = (
  metaState: unknown,
  framesSinceLastSeen: number,
): MemoryRecordView['state'] => {
  if (typeof metaState === 'string') {
    const normalized = metaState.toLowerCase();
    if (normalized.includes('prune')) return 'Pruned';
    if (normalized.includes('dorm')) return 'Dormant';
    if (normalized.includes('active')) return 'Active';
  }
  if (framesSinceLastSeen <= 10) return 'Active';
  if (framesSinceLastSeen <= 60) return 'Dormant';
  return 'Pruned';
};

const computeStabilityScore = (
  provided: number | undefined,
  observations: number,
  framesSinceLastSeen: number,
) => {
  if (typeof provided === 'number' && Number.isFinite(provided)) {
    return clamp01(provided);
  }
  const observationStrength = Math.min(1, observations / 50);
  const freshness = Math.max(0, 1 - framesSinceLastSeen / 90);
  return clamp01(observationStrength * 0.65 + freshness * 0.35);
};

export function LongTermMemory() {
  const [selectedEntity, setSelectedEntity] = useState<number | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [stateFilter, setStateFilter] = useState<MemoryRecordView['state'] | 'All'>('All');
  const { metadata: streamingMetadata } = useStreaming();

  const metadata = streamingMetadata ?? EMPTY_METADATA;

  const memoryRecords = useMemo<MemoryRecordView[]>(() => {
    const frameIdx = metadata.frameIdx ?? 0;

    return metadata.entityRecords.map((entity, idx) => {
      const trackId =
        entity.trackHistory && entity.trackHistory.length > 0
          ? entity.trackHistory[entity.trackHistory.length - 1]
          : entity.entityId ?? idx;

      const firstSeenFrame =
        Array.isArray(entity.seenFrames) && entity.seenFrames.length > 0
          ? entity.seenFrames[0]
          : undefined;
      const lastSeenFrame =
        typeof entity.lastSeenFrame === 'number'
          ? entity.lastSeenFrame
          : Array.isArray(entity.seenFrames) && entity.seenFrames.length > 0
            ? entity.seenFrames[entity.seenFrames.length - 1]
            : undefined;

      const framesSinceLastSeen =
        lastSeenFrame !== undefined && frameIdx > 0 ? Math.max(0, frameIdx - lastSeenFrame) : 0;

      const { short: firstSeenShort, long: firstSeenLong } = buildTimeLabels(
        readMetaField(entity.meta, ['firstSeenAt', 'first_seen_at', 'firstSeen', 'first_seen']),
        firstSeenFrame,
      );

      const { short: lastSeenShort, long: lastSeenLong } = buildTimeLabels(
        readMetaField(entity.meta, ['lastSeenAt', 'last_seen_at', 'lastSeen', 'last_seen']),
        lastSeenFrame ?? entity.lastSeenFrame,
      );

      const observations = Array.isArray(entity.seenFrames) ? entity.seenFrames.length : 0;
      const stabilityScore = computeStabilityScore(
        readMetaField<number>(entity.meta, ['stability', 'stabilityScore', 'score']),
        observations,
        framesSinceLastSeen,
      );

      const lastPosition =
        readMetaField<string>(entity.meta, ['lastPosition', 'last_position', 'position']) ??
        describeBoxPosition(entity.lastBox);

      const state = deriveState(readMetaField(entity.meta, ['state', 'status']), framesSinceLastSeen);

      return {
        entityId: entity.entityId ?? idx,
        trackId,
        cls: entity.baseCls || 'entity',
        firstSeenShort,
        firstSeenLong,
        lastSeenShort,
        lastSeenLong,
        totalObservations: observations,
        stabilityScore,
        lastPosition,
        state,
        avatar: pickColor(entity.entityId, trackId, idx),
      };
    });
  }, [metadata]);

  const filteredRecords = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    return memoryRecords.filter((record) => {
      if (stateFilter !== 'All' && record.state !== stateFilter) return false;
      if (!term) return true;
      return (
        record.cls.toLowerCase().includes(term) ||
        record.entityId.toString().includes(term) ||
        record.trackId.toString().includes(term) ||
        record.lastPosition.toLowerCase().includes(term)
      );
    });
  }, [memoryRecords, searchTerm, stateFilter]);

  useEffect(() => {
    if (filteredRecords.length === 0) {
      setSelectedEntity(null);
      return;
    }
    const alreadySelected = filteredRecords.some((r) => r.entityId === selectedEntity);
    if (!alreadySelected) {
      setSelectedEntity(filteredRecords[0].entityId);
    }
  }, [filteredRecords, selectedEntity]);

  const selected = filteredRecords.find((e) => e.entityId === selectedEntity) ?? null;
  const hasRecords = metadata.entityRecords.length > 0;

  return (
    <div className="flex h-full flex-col space-y-6">
      {/* Subtitle */}
      <p className="text-[14px] text-[#a0aec0]">
        Entity records stored in long-term memory (live from streaming metadata)
      </p>

      {/* Two Column Layout */}
      <div className="grid h-full min-h-0 grid-cols-[1fr_400px] gap-6">
        {/* LEFT COLUMN - Entity Table */}
        <div className="flex h-full flex-col rounded-[15px] bg-white p-6 shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[18px] text-[#2d3748]">Entity Long-Term Memory</h3>
            <div className="rounded-lg bg-gray-50 px-3 py-1 text-[11px] uppercase tracking-wide text-[#a0aec0]">
              {metadata.entityRecords.length} records
            </div>
          </div>

          {/* Controls Row */}
          <div className="mb-4 flex items-center gap-3">
            {/* Search */}
            <div className="flex flex-1 items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-3 py-2">
              <Search className="size-4 text-[#a0aec0]" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search by ID, label, or state"
                className="flex-1 border-0 bg-transparent text-[14px] text-[#2d3748] outline-none"
              />
            </div>

            {/* Sort Dropdown */}
            <button className="flex items-center gap-2 rounded-lg border border-[#e2e8f0] bg-white px-4 py-2 text-[14px] text-[#2d3748] transition-colors hover:bg-gray-50">
              <span>Last seen</span>
              <ChevronDown className="size-4" />
            </button>
          </div>

          {/* Filter Chips */}
          <div className="mb-4 flex flex-wrap gap-2">
            {(['All', 'Active', 'Dormant', 'Pruned'] as const).map((state) => (
              <button
                key={state}
                onClick={() => setStateFilter(state)}
                className={`rounded-lg px-3 py-1 text-[12px] transition-colors ${
                  stateFilter === state
                    ? 'bg-[#4fd1c5] text-white'
                    : 'bg-gray-100 text-[#a0aec0] hover:bg-gray-200'
                }`}
              >
                {state}
              </button>
            ))}
          </div>

          {/* Table Header */}
          <div className="mb-3 grid grid-cols-[80px_1fr_80px_120px_120px_100px_100px_80px] gap-3 border-b border-[#e2e8f0] pb-3">
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">ID</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Class</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Track</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">First Seen</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Last Seen</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Obs.</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">Stability</div>
            <div className="text-[10px] uppercase tracking-wide text-[#a0aec0]">State</div>
          </div>

          {/* Table Rows */}
          <div className="flex-1 space-y-2 overflow-y-auto">
            {filteredRecords.length === 0 ? (
              <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-[#e2e8f0] bg-gray-50/60 p-6 text-center text-[13px] text-[#a0aec0]">
                {hasRecords
                  ? 'No records match the current filters.'
                  : 'No entity records received yet. Waiting for streaming metadata...'}
              </div>
            ) : (
              filteredRecords.map((entity) => (
                <div
                  key={entity.entityId}
                  onClick={() => setSelectedEntity(entity.entityId)}
                  className={`grid grid-cols-[80px_1fr_80px_120px_120px_100px_100px_80px] cursor-pointer items-center gap-3 rounded-lg px-3 py-3 transition-colors ${
                    selectedEntity === entity.entityId
                      ? 'border border-[#4fd1c5] bg-[#4fd1c5]/10'
                      : 'border border-transparent hover:bg-gray-50'
                  }`}
                >
                  {/* Entity ID */}
                  <div className="text-[14px] text-[#2d3748]">#{entity.entityId}</div>

                  {/* Class with Avatar */}
                  <div className="flex items-center gap-2">
                    <div
                      className="flex size-8 items-center justify-center rounded-full text-center text-[11px] text-white"
                      style={{ backgroundColor: entity.avatar }}
                    >
                      {entity.cls[0]?.toUpperCase?.() ?? '?'}
                    </div>
                    <div className="text-[14px] text-[#2d3748] capitalize">{entity.cls}</div>
                  </div>

                  {/* Track ID */}
                  <div className="text-[14px] text-[#a0aec0]">#{entity.trackId}</div>

                  {/* First Seen */}
                  <div className="text-[12px] text-[#a0aec0]">{entity.firstSeenShort}</div>

                  {/* Last Seen */}
                  <div className="text-[12px] text-[#a0aec0]">{entity.lastSeenShort}</div>

                  {/* Observations */}
                  <div className="text-[14px] text-[#2d3748]">
                    {entity.totalObservations.toLocaleString()}
                  </div>

                  {/* Stability */}
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

                  {/* State */}
                  <div>
                    <span
                      className={`inline-block rounded-md px-2 py-1 text-[11px] text-white ${
                        entity.state === 'Active'
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

        {/* RIGHT COLUMN - Entity Detail Panel */}
        <div className="flex h-full flex-col overflow-y-auto rounded-[15px] bg-white p-6 shadow-[0px_3.5px_5.5px_0px_rgba(0,0,0,0.02)]">
          <h3 className="mb-4 text-[18px] text-[#2d3748]">Selected Entity</h3>

          {!selected ? (
            // Empty State
            <div className="flex flex-1 flex-col items-center justify-center text-center">
              <div className="mb-4 flex size-16 items-center justify-center rounded-full bg-gray-100">
                <FileText className="size-8 text-[#a0aec0]" />
              </div>
              <p className="text-[14px] text-[#a0aec0]">
                {hasRecords
                  ? 'Choose a record on the left to inspect long-term details.'
                  : 'Waiting for entity records from the stream...'}
              </p>
            </div>
          ) : (
            // Selected Entity Details
            <div className="space-y-6">
              {/* Entity Header */}
              <div className="rounded-lg bg-gradient-to-br from-[#4fd1c5]/10 to-[#38b2ac]/10 p-4">
                <div className="mb-2 flex items-center gap-3">
                  <div
                    className="flex size-12 items-center justify-center rounded-full text-[18px] text-white"
                    style={{ backgroundColor: selected.avatar }}
                  >
                    {selected.entityId}
                  </div>
                  <div>
                    <div className="text-[18px] text-[#2d3748]">
                      Entity #{selected.entityId} - {selected.cls}
                    </div>
                    <div className="text-[12px] text-[#a0aec0]">Track ID #{selected.trackId}</div>
                  </div>
                </div>
              </div>

              {/* Metadata Grid */}
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-[#a0aec0]">
                    First Seen
                  </div>
                  <div className="text-[14px] text-[#2d3748]">{selected.firstSeenLong}</div>
                </div>

                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-[#a0aec0]">
                    Last Seen
                  </div>
                  <div className="text-[14px] text-[#2d3748]">{selected.lastSeenLong}</div>
                </div>

                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-[#a0aec0]">
                    Observations
                  </div>
                  <div className="text-[20px] text-[#2d3748]">
                    {selected.totalObservations.toLocaleString()}
                  </div>
                </div>

                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-[#a0aec0]">
                    State
                  </div>
                  <span
                    className={`inline-block rounded-md px-3 py-1 text-[12px] text-white ${
                      selected.state === 'Active'
                        ? 'bg-[#4fd1c5]'
                        : selected.state === 'Dormant'
                          ? 'bg-[#4299e1]'
                          : 'bg-[#fc8181]'
                    }`}
                  >
                    {selected.state}
                  </span>
                </div>
              </div>

              {/* Stability Section */}
              <div className="rounded-lg border border-[#e2e8f0] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <div className="text-[14px] text-[#2d3748]">Stability Score</div>
                  <div className="text-[24px] text-[#4fd1c5]">
                    {selected.stabilityScore.toFixed(2)}
                  </div>
                </div>
                <div className="mb-2">
                  <div className="h-2 w-full rounded-full bg-[#e2e8f0]">
                    <div
                      className="h-2 rounded-full bg-[#4fd1c5]"
                      style={{ width: `${selected.stabilityScore * 100}%` }}
                    ></div>
                  </div>
                </div>
                <div className="text-[12px] text-[#a0aec0]">
                  {selected.stabilityScore >= 0.8
                    ? 'Stable'
                    : selected.stabilityScore >= 0.5
                      ? 'Moderate'
                      : 'Unstable'}
                </div>
              </div>

              {/* Lifetime Timeline */}
              <div className="rounded-lg border border-[#e2e8f0] p-4">
                <div className="mb-3 text-[14px] text-[#2d3748]">Lifetime Timeline</div>
                <div className="relative h-2 w-full rounded-full bg-[#e2e8f0]">
                  <div
                    className="absolute h-2 rounded-l-full bg-gradient-to-r from-[#4fd1c5] to-[#38b2ac]"
                    style={{ width: '75%' }}
                  ></div>
                  {/* Timeline dots */}
                  <div className="absolute left-[25%] top-1/2 size-3 -translate-y-1/2 rounded-full border-2 border-white bg-[#4fd1c5]"></div>
                  <div className="absolute left-[50%] top-1/2 size-3 -translate-y-1/2 rounded-full border-2 border-white bg-[#4fd1c5]"></div>
                  <div className="absolute left-[75%] top-1/2 size-3 -translate-y-1/2 rounded-full border-2 border-white bg-[#38b2ac]"></div>
                </div>
                <div className="mt-2 flex justify-between text-[10px] text-[#a0aec0]">
                  <span>First</span>
                  <span>Active Period</span>
                  <span>Current</span>
                </div>
              </div>

              {/* Position & Summary */}
              <div className="rounded-lg bg-gray-50 p-4">
                <div className="mb-2 text-[12px] uppercase tracking-wide text-[#a0aec0]">
                  Last Position
                </div>
                <div className="mb-3 text-[14px] text-[#2d3748] capitalize">
                  {selected.lastPosition}
                </div>
                <div className="text-[13px] leading-relaxed text-[#a0aec0]">
                  {selected.state === 'Active'
                    ? `Long-lived active ${selected.cls} track. Mostly appears ${selected.lastPosition}. High stability indicates consistent detection.`
                    : selected.state === 'Dormant'
                      ? `${selected.cls} track is dormant. Last observed ${selected.lastPosition}. May reactivate if entity reappears.`
                      : `${selected.cls} track has been pruned from active memory due to extended absence. Last known ${selected.lastPosition}.`}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
