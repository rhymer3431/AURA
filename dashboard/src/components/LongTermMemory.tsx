import { useEffect, useMemo, useState } from 'react';
import { useStreaming } from '../application/streaming/StreamingProvider';
import type { StreamEntityRecord } from '../domain/streaming/streamTypes';
import { EMPTY_METADATA } from '../infra/streaming/metadataParsers';
import { EntityDetailPanel } from './longTermMemory/EntityDetailPanel';
import { MemoryRecordsPanel } from './longTermMemory/MemoryRecordsPanel';
import type { MemoryRecordView } from './longTermMemory/types';

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
  if (framesSinceLastSeen <= 30) return 'Dormant';
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
  const entityRecords = metadata.entityRecords ?? [];

  const memoryRecords = useMemo<MemoryRecordView[]>(() => {
    const frameIdx = metadata.frameIdx ?? 0;

    return entityRecords.map((entity, idx) => {
      const trackId =
        entity.trackHistory && entity.trackHistory.length > 0
          ? entity.trackHistory[entity.trackHistory.length - 1]
          : entity.entityId ?? idx;

      const lastSeenFrame = typeof entity.lastSeenFrame === 'number' ? entity.lastSeenFrame : 0;
      const seenCount = typeof entity.seenCount === 'number' ? entity.seenCount : 0;

      const metaFirstSeen = readMetaField<number>(entity.meta, [
        'firstSeenFrame',
        'first_seen_frame',
        'firstSeen',
        'first_seen',
      ]);
      const firstSeenFrame =
        typeof metaFirstSeen === 'number'
          ? metaFirstSeen
          : lastSeenFrame && seenCount > 0
            ? Math.max(0, lastSeenFrame - seenCount + 1)
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

      const observations = seenCount;
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
        lastSeenFrame: lastSeenFrame ?? 0,
        seenCount: seenCount,
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
  }, [entityRecords, metadata.frameIdx]);

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
  const hasRecords = entityRecords.length > 0;

  return (
    <div className="flex h-full flex-col space-y-6 pt-4">

      {/* Two Column Layout */}
      <div className="grid h-full min-h-0 grid-cols-[1fr_400px] gap-6">
        <MemoryRecordsPanel
          records={filteredRecords}
          searchTerm={searchTerm}
          onSearchTermChange={setSearchTerm}
          stateFilter={stateFilter}
          onStateFilterChange={setStateFilter}
          selectedEntityId={selectedEntity}
          onSelectEntity={(entityId) => setSelectedEntity(entityId)}
          entityCount={entityRecords.length}
          hasRecords={hasRecords}
        />
        <EntityDetailPanel selectedRecord={selected} hasRecords={hasRecords} />
      </div>
    </div>
  );
}
