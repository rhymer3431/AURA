import type { StreamEntity } from '../streaming/streamTypes';

export type FrameSize = { width: number; height: number }

export type BoundingBoxOverlay = {
  id: string
  label: string
  confidence?: number
  color: string
  box: {
    leftPct: number
    topPct: number
    widthPct: number
    heightPct: number
  }
}

const COLOR_PALETTE = [
  '#4fd1c5',
  '#22c55e',
  '#3b82f6',
  '#f59e0b',
  '#ef4444',
  '#a855f7',
  '#06b6d4',
  '#ec4899',
]

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

const pickColor = (entity: StreamEntity, idx: number) => {
  const key = entity.trackId ?? entity.entityId ?? idx
  const paletteIdx = Math.abs(key) % COLOR_PALETTE.length
  return COLOR_PALETTE[paletteIdx]
}

const normalizeBox = (box: number[], frameSize?: FrameSize) => {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box
  const isPixelSpace = frameSize && Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) > 1

  const nx1 = isPixelSpace && frameSize ? x1 / frameSize.width : x1
  const ny1 = isPixelSpace && frameSize ? y1 / frameSize.height : y1
  const nx2 = isPixelSpace && frameSize ? x2 / frameSize.width : x2
  const ny2 = isPixelSpace && frameSize ? y2 / frameSize.height : y2

  const left = clamp01(Math.min(nx1, nx2))
  const right = clamp01(Math.max(nx1, nx2))
  const top = clamp01(Math.min(ny1, ny2))
  const bottom = clamp01(Math.max(ny1, ny2))

  return {
    leftPct: left,
    topPct: top,
    widthPct: clamp01(right - left),
    heightPct: clamp01(bottom - top),
  }
}

export function buildBoundingBoxOverlays(
  entities: StreamEntity[],
  frameSize?: FrameSize,
): BoundingBoxOverlay[] {
  return entities
    .filter((entity) => Array.isArray(entity.box) && entity.box.length >= 4)
    .map((entity, idx) => ({
      id: `bbox-${entity.entityId ?? idx}`,
      label: `${entity.cls ?? 'Entity'} #${entity.entityId ?? entity.trackId ?? idx}`,
      confidence: typeof entity.score === 'number' ? entity.score : undefined,
      color: pickColor(entity, idx),
      box: normalizeBox(entity.box, frameSize),
    }))
    .filter((overlay) => overlay.box.widthPct > 0 && overlay.box.heightPct > 0)
}
