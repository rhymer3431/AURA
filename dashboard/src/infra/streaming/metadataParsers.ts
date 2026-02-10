import type {
  MetadataMessage,
  StreamMetadata,
  StreamOctomapCloud,
  StreamProjectedMap,
  StreamProjectedMapPreview,
  StreamSemanticMap,
  StreamSlamPose,
  StreamSource,
} from '../../domain/streaming/streamTypes'
import { parseSceneGraphDiff } from './parseSceneGraphDiff'

export const EMPTY_METADATA: StreamMetadata = {
  frameIdx: 0,
  caption: '',
  focusTargets: [],
  entities: [],
  entityRecords: [],
  relations: [],
  sceneGraphDiff: undefined,
  streamSource: undefined,
  slamPoseTopic: undefined,
  slamPose: undefined,
  semanticMap: undefined,
}

function asFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function parseStreamSource(raw: unknown): StreamSource | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const src = raw as Record<string, unknown>
  return {
    type: typeof src.type === 'string' ? src.type : undefined,
    imageTopic: typeof src.imageTopic === 'string' ? src.imageTopic : undefined,
    videoPath: typeof src.videoPath === 'string' ? src.videoPath : undefined,
  }
}

function parseSlamPose(raw: unknown): StreamSlamPose | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const pose = raw as Record<string, unknown>
  const pos = pose.position as Record<string, unknown> | undefined
  const ori = pose.orientation as Record<string, unknown> | undefined
  if (!pos || !ori) return undefined

  const px = asFiniteNumber(pos.x)
  const py = asFiniteNumber(pos.y)
  const pz = asFiniteNumber(pos.z)
  const ox = asFiniteNumber(ori.x)
  const oy = asFiniteNumber(ori.y)
  const oz = asFiniteNumber(ori.z)
  const ow = asFiniteNumber(ori.w)
  if (px === null || py === null || pz === null || ox === null || oy === null || oz === null || ow === null) {
    return undefined
  }

  const stamp = pose.stamp as Record<string, unknown> | undefined
  const sec = stamp ? asFiniteNumber(stamp.sec) : null
  const nanosec = stamp ? asFiniteNumber(stamp.nanosec) : null

  return {
    position: { x: px, y: py, z: pz },
    orientation: { x: ox, y: oy, z: oz, w: ow },
    stamp: sec !== null && nanosec !== null ? { sec, nanosec } : undefined,
    frameId: typeof pose.frameId === 'string' ? pose.frameId : undefined,
  }
}

function parseProjectedMapPreview(
  raw: unknown,
  previous?: StreamProjectedMapPreview,
): StreamProjectedMapPreview | undefined {
  if (!raw || typeof raw !== 'object') {
    return previous
  }
  const src = raw as Record<string, unknown>
  const width = asFiniteNumber(src.width)
  const height = asFiniteNumber(src.height)
  const rows = Array.isArray(src.rows) ? src.rows.map(String) : []
  if (width === null || height === null || rows.length === 0) {
    return previous
  }

  const encoding = typeof src.encoding === 'string' ? src.encoding : 'ufo-v1'
  const revision = asFiniteNumber(src.revision)
  return {
    encoding,
    width,
    height,
    rows,
    revision: revision === null ? previous?.revision : revision,
  }
}

function parseProjectedMap(
  raw: unknown,
  previous?: StreamProjectedMap,
): StreamProjectedMap | undefined {
  if (!raw || typeof raw !== 'object') {
    return previous
  }
  const src = raw as Record<string, unknown>
  const width = asFiniteNumber(src.width)
  const height = asFiniteNumber(src.height)
  const resolution = asFiniteNumber(src.resolution)
  const occupiedCells = asFiniteNumber(src.occupiedCells)
  const freeCells = asFiniteNumber(src.freeCells)
  const unknownCells = asFiniteNumber(src.unknownCells)
  const knownRatio = asFiniteNumber(src.knownRatio)

  if (
    width === null ||
    height === null ||
    resolution === null ||
    occupiedCells === null ||
    freeCells === null ||
    unknownCells === null ||
    knownRatio === null
  ) {
    return previous
  }

  const stampRaw = src.stamp as Record<string, unknown> | undefined
  const sec = stampRaw ? asFiniteNumber(stampRaw.sec) : null
  const nanosec = stampRaw ? asFiniteNumber(stampRaw.nanosec) : null

  const previewRevision = asFiniteNumber(src.previewRevision)
  const preview = parseProjectedMapPreview(src.preview, previous?.preview)
  return {
    width,
    height,
    resolution,
    occupiedCells,
    freeCells,
    unknownCells,
    knownRatio,
    frameId: typeof src.frameId === 'string' ? src.frameId : previous?.frameId,
    stamp:
      sec !== null && nanosec !== null
        ? { sec, nanosec }
        : previous?.stamp,
    previewRevision: previewRevision === null ? previous?.previewRevision : previewRevision,
    preview,
  }
}

function parseOctomapCloud(
  raw: unknown,
  previous?: StreamOctomapCloud,
): StreamOctomapCloud | undefined {
  if (!raw || typeof raw !== 'object') {
    return previous
  }
  const src = raw as Record<string, unknown>
  const pointCount = asFiniteNumber(src.pointCount)
  if (pointCount === null) {
    return previous
  }
  const stampRaw = src.stamp as Record<string, unknown> | undefined
  const sec = stampRaw ? asFiniteNumber(stampRaw.sec) : null
  const nanosec = stampRaw ? asFiniteNumber(stampRaw.nanosec) : null
  return {
    pointCount,
    frameId: typeof src.frameId === 'string' ? src.frameId : previous?.frameId,
    stamp:
      sec !== null && nanosec !== null
        ? { sec, nanosec }
        : previous?.stamp,
  }
}

function parseSemanticMap(
  raw: unknown,
  previous?: StreamSemanticMap,
): StreamSemanticMap | undefined {
  if (!raw || typeof raw !== 'object') {
    return previous
  }
  const src = raw as Record<string, unknown>
  return {
    projectedMapTopic:
      typeof src.projectedMapTopic === 'string'
        ? src.projectedMapTopic
        : previous?.projectedMapTopic,
    octomapCloudTopic:
      typeof src.octomapCloudTopic === 'string'
        ? src.octomapCloudTopic
        : previous?.octomapCloudTopic,
    projectedMap: parseProjectedMap(src.projectedMap, previous?.projectedMap),
    octomapCloud: parseOctomapCloud(src.octomapCloud, previous?.octomapCloud),
  }
}

export function parseMetadataMessage(raw: unknown, previous?: StreamMetadata): StreamMetadata {
  if (!raw || typeof raw !== 'object') return { ...EMPTY_METADATA }
  const msg = raw as MetadataMessage
  const prev = previous ?? EMPTY_METADATA

  return {
    frameIdx: typeof msg.frameIdx === 'number' ? msg.frameIdx : EMPTY_METADATA.frameIdx,
    caption: typeof msg.caption === 'string' ? msg.caption : EMPTY_METADATA.caption,
    focusTargets: Array.isArray(msg.focusTargets)
      ? msg.focusTargets.map(String)
      : [],
    entities: Array.isArray(msg.entities) ? msg.entities : [],
    entityRecords: Array.isArray(msg.entityRecords) ? msg.entityRecords : [],
    relations: Array.isArray(msg.relations) ? msg.relations : [],
    sceneGraphDiff: parseSceneGraphDiff(msg.sceneGraphDiff as any),
    streamSource: parseStreamSource(msg.streamSource),
    slamPoseTopic: typeof msg.slamPoseTopic === 'string' ? msg.slamPoseTopic : undefined,
    slamPose: parseSlamPose(msg.slamPose),
    semanticMap: parseSemanticMap((msg as Record<string, unknown>).semanticMap, prev.semanticMap),
  }
}
