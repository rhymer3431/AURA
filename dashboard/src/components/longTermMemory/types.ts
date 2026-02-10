export type MemoryRecordView = {
  entityId: number;
  trackId: number;
  cls: string;
  lastSeenFrame: number;
  seenCount: number;
  totalObservations: number;
  firstSeenShort: string;
  firstSeenLong: string;
  lastSeenShort: string;
  lastSeenLong: string;
  stabilityScore: number;
  lastPosition: string;
  state: 'Active' | 'Dormant' | 'Pruned';
  avatar: string;
};
