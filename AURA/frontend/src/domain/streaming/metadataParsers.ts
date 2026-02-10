import type { MetadataMessage, StreamMetadata } from './streamTypes'

export const EMPTY_METADATA: StreamMetadata = {
  frameIdx: 0,
  caption: '',
  focusTargets: [],
  entities: [],
  entityRecords: [],
  relations: [],
}

export function parseMetadataMessage(raw: unknown): StreamMetadata {
  if (!raw || typeof raw !== 'object') return { ...EMPTY_METADATA }
  const msg = raw as MetadataMessage
  return {
    frameIdx: typeof msg.frameIdx === 'number' ? msg.frameIdx : EMPTY_METADATA.frameIdx,
    caption: typeof msg.caption === 'string' ? msg.caption : EMPTY_METADATA.caption,
    focusTargets: Array.isArray(msg.focusTargets)
      ? msg.focusTargets.map(String)
      : [],
    entities: Array.isArray(msg.entities) ? msg.entities : [],
    entityRecords: Array.isArray(msg.entityRecords) ? msg.entityRecords : [],
    relations: Array.isArray(msg.relations) ? msg.relations : [],
  }
}
