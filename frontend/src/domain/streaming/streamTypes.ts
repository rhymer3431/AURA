// src/domain/streaming/streamTypes.ts

export type StreamState = 'idle' | 'connecting' | 'playing' | 'paused' | 'error'

export type StreamEntity = {
  entityId: number
  trackId: number
  cls: string
  box: number[]
  score: number
}

export type StreamEntityRecord = {
  entityId: number
  baseCls: string
  lastBox: number[]
  lastSeenFrame: number
  seenFrames: number[]
  trackHistory: number[]
  meta?: Record<string, unknown>
}

export type StreamRelation = {
  subjectEntityId: number
  objectEntityId: number
  relation: string
  relationId?: number
  confidence?: number
  type?: 'static' | 'temporal'
  subjectCls?: string
  objectCls?: string
}

export type StreamMetadata = {
  frameIdx: number
  caption: string
  focusTargets: string[]
  entities: StreamEntity[]
  entityRecords: StreamEntityRecord[]
  relations?: StreamRelation[]
}

export type InitMessage = {
  type: 'init'
  runtimeReady: boolean
  llmModel: string
  device: string
}

export type MetadataMessage = StreamMetadata & { type?: 'metadata' }

export type StreamError = {
  message: string
}
