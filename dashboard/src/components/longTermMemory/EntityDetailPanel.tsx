import { FileText } from 'lucide-react';
import type { MemoryRecordView } from './types';

export type EntityDetailPanelProps = {
  selectedRecord: MemoryRecordView | null;
  hasRecords: boolean;
};

export function EntityDetailPanel({ selectedRecord, hasRecords }: EntityDetailPanelProps) {
  if (!selectedRecord) {
    const emptyMessage = hasRecords
      ? 'Choose a record on the left to inspect long-term details.'
      : 'Waiting for entity records from the stream...';

    return (
      <div className="panel overflow-y-auto">
        <h3 className="mb-4 text-[18px] text-[#2d3748]">Selected Entity</h3>
        <div className="flex flex-1 flex-col items-center justify-center text-center">
          <div className="mb-4 flex size-16 items-center justify-center rounded-full bg-gray-100">
            <FileText className="size-8 text-[#a0aec0]" />
          </div>
          <p className="text-[14px] text-[#a0aec0]">{emptyMessage}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel overflow-y-auto">
      <h3 className="mb-4 text-[18px] text-[#2d3748]">Selected Entity</h3>
      <div className="space-y-6">
        <div className="rounded-lg bg-gradient-to-br from-[#4fd1c5]/10 to-[#38b2ac]/10 p-4">
          <div className="mb-2 flex items-center gap-3">
            <div
              className="flex size-12 items-center justify-center rounded-full text-[18px] text-white"
              style={{ backgroundColor: selectedRecord.avatar }}
            >
              {selectedRecord.entityId}
            </div>
            <div>
              <div className="text-[18px] text-[#2d3748]">
                Entity #{selectedRecord.entityId} - {selectedRecord.cls}
              </div>
              <div className="text-[12px] text-[#a0aec0]">Track ID #{selectedRecord.trackId}</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg bg-gray-50 p-3">
            <div className="mb-1 text-[10px] uppercase tracking-wide text-[#a0aec0]">
              First Seen
            </div>
            <div className="text-[14px] text-[#2d3748]">{selectedRecord.firstSeenLong}</div>
          </div>
          <div className="rounded-lg bg-gray-50 p-3">
            <div className="mb-1 text-[10px] uppercase tracking-wide text-[#a0aec0]">
              Last Seen
            </div>
            <div className="text-[14px] text-[#2d3748]">{selectedRecord.lastSeenLong}</div>
          </div>
          <div className="rounded-lg bg-gray-50 p-3">
            <div className="mb-1 text-[10px] uppercase tracking-wide text-[#a0aec0]">
              Observations
            </div>
            <div className="text-[20px] text-[#2d3748]">
              {selectedRecord.totalObservations.toLocaleString()}
            </div>
          </div>
          <div className="rounded-lg bg-gray-50 p-3">
            <div className="mb-1 text-[10px] uppercase tracking-wide text-[#a0aec0]">
              State
            </div>
            <span
              className={`inline-block rounded-md px-3 py-1 text-[12px] text-white ${
                selectedRecord.state === 'Active'
                  ? 'bg-[#4fd1c5]'
                  : selectedRecord.state === 'Dormant'
                    ? 'bg-[#4299e1]'
                    : 'bg-[#fc8181]'
              }`}
            >
              {selectedRecord.state}
            </span>
          </div>
        </div>

        <div className="rounded-lg border border-[#e2e8f0] p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-[14px] text-[#2d3748]">Stability Score</div>
            <div className="text-[24px] text-[#4fd1c5]">
              {selectedRecord.stabilityScore.toFixed(2)}
            </div>
          </div>
          <div className="mb-2">
            <div className="h-2 w-full rounded-full bg-[#e2e8f0]">
              <div
                className="h-2 rounded-full bg-[#4fd1c5]"
                style={{ width: `${selectedRecord.stabilityScore * 100}%` }}
              ></div>
            </div>
          </div>
          <div className="text-[12px] text-[#a0aec0]">
            {selectedRecord.stabilityScore >= 0.8
              ? 'Stable'
              : selectedRecord.stabilityScore >= 0.5
                ? 'Moderate'
                : 'Unstable'}
          </div>
        </div>

        <div className="rounded-lg bg-gray-50 p-4">
          <div className="mb-2 text-[12px] uppercase tracking-wide text-[#a0aec0]">
            Last Position
          </div>
          <div className="mb-3 text-[14px] text-[#2d3748] capitalize">
            {selectedRecord.lastPosition}
          </div>
          <div className="text-[13px] leading-relaxed text-[#a0aec0]">
            {selectedRecord.state === 'Active'
              ? `Long-lived active ${selectedRecord.cls} track. Mostly appears ${selectedRecord.lastPosition}. High stability indicates consistent detection.`
              : selectedRecord.state === 'Dormant'
                ? `${selectedRecord.cls} track is dormant. Last observed ${selectedRecord.lastPosition}. May reactivate if entity reappears.`
                : `${selectedRecord.cls} track has been pruned from active memory due to extended absence. Last known ${selectedRecord.lastPosition}.`}
          </div>
        </div>
      </div>
    </div>
  );
}
