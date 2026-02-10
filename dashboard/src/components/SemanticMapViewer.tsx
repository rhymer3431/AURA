import { useEffect, useMemo, useRef } from 'react'
import { useStreaming } from '../application/streaming/StreamingProvider'
import type { StreamProjectedMapPreview } from '../domain/streaming/streamTypes'
import { EMPTY_METADATA } from '../infra/streaming/metadataParsers'

const COLOR_UNKNOWN = [237, 242, 247] as const
const COLOR_FREE = [198, 246, 213] as const
const COLOR_OCCUPIED = [252, 129, 129] as const

function drawProjectedMapPreview(
  canvas: HTMLCanvasElement,
  preview: StreamProjectedMapPreview,
) {
  const width = Math.max(1, Math.floor(preview.width))
  const height = Math.max(1, Math.floor(preview.height))

  canvas.width = width
  canvas.height = height

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const imageData = ctx.createImageData(width, height)
  const data = imageData.data
  const rows = Array.isArray(preview.rows) ? preview.rows : []

  for (let y = 0; y < height; y += 1) {
    const row = rows[y] ?? ''
    for (let x = 0; x < width; x += 1) {
      const code = row.charCodeAt(x)
      let color = COLOR_UNKNOWN
      if (code === 102) color = COLOR_FREE // 'f'
      if (code === 111) color = COLOR_OCCUPIED // 'o'
      const idx = (y * width + x) * 4
      data[idx + 0] = color[0]
      data[idx + 1] = color[1]
      data[idx + 2] = color[2]
      data[idx + 3] = 255
    }
  }

  ctx.putImageData(imageData, 0, 0)
}

function toPercent(value: number | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '0.0%'
  return `${(value * 100).toFixed(1)}%`
}

export function SemanticMapViewer() {
  const {
    streamState,
    fps,
    metadata: rawMetadata,
    connect,
  } = useStreaming()

  const metadata = rawMetadata ?? EMPTY_METADATA
  const semanticMap = metadata.semanticMap
  const projectedMap = semanticMap?.projectedMap
  const octomapCloud = semanticMap?.octomapCloud
  const preview = projectedMap?.preview
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    if (!preview || !Array.isArray(preview.rows) || preview.rows.length === 0) {
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      canvas.width = 1
      canvas.height = 1
      ctx.fillStyle = '#f8fafc'
      ctx.fillRect(0, 0, 1, 1)
      return
    }
    drawProjectedMapPreview(canvas, preview)
  }, [preview])

  const projectedSummary = useMemo(() => {
    if (!projectedMap) {
      return {
        size: 'n/a',
        resolution: 'n/a',
        known: '0.0%',
      }
    }
    return {
      size: `${projectedMap.width} x ${projectedMap.height}`,
      resolution: `${projectedMap.resolution.toFixed(3)} m/cell`,
      known: toPercent(projectedMap.knownRatio),
    }
  }, [projectedMap])

  return (
    <div className="flex h-full flex-col space-y-6 pt-4">
      <p className="text-[14px] text-[#a0aec0]">
        Projected semantic occupancy map preview and octomap cloud status.
      </p>

      <div className="grid h-full min-h-0 grid-cols-[1.35fr_1fr] gap-6">
        <div className="panel flex min-h-0 flex-col">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[18px] text-[#2d3748]">Semantic Projected Map</h3>
            <div className="flex items-center gap-2">
              <div className="rounded-lg bg-[#4fd1c5] px-3 py-1">
                <span className="text-[12px] text-white">{fps.toFixed(1)} FPS</span>
              </div>
              <button
                onClick={connect}
                className="rounded-lg border border-[#e2e8f0] bg-white px-3 py-1 text-[12px] text-[#2d3748] transition-colors hover:bg-gray-50"
              >
                Connect
              </button>
            </div>
          </div>

          <div className="relative flex-1 overflow-hidden rounded-xl border border-[#e2e8f0] bg-gradient-to-br from-gray-50 to-white">
            <canvas
              ref={canvasRef}
              className="h-full w-full [image-rendering:pixelated]"
            />
            {!preview && (
              <div className="absolute inset-0 flex items-center justify-center text-[14px] text-[#a0aec0]">
                Waiting for /semantic_map/projected_map...
              </div>
            )}
          </div>

          <div className="mt-3 flex items-center gap-4 text-[11px] text-[#a0aec0]">
            <span>state: {streamState}</span>
            <span>frame: {metadata.frameIdx}</span>
            <span>preview: {preview ? `${preview.width}x${preview.height}` : 'n/a'}</span>
          </div>
        </div>

        <div className="flex min-h-0 flex-col gap-6">
          <div className="panel">
            <h3 className="mb-3 text-[18px] text-[#2d3748]">Projected Map Stats</h3>
            {projectedMap ? (
              <div className="space-y-3 text-[13px] text-[#2d3748]">
                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Geometry</div>
                  <div>size: {projectedSummary.size}</div>
                  <div>resolution: {projectedSummary.resolution}</div>
                  <div>known area: {projectedSummary.known}</div>
                </div>
                <div className="rounded-lg bg-gray-50 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Cells</div>
                  <div>occupied: {projectedMap.occupiedCells}</div>
                  <div>free: {projectedMap.freeCells}</div>
                  <div>unknown: {projectedMap.unknownCells}</div>
                </div>
                <div className="rounded-lg bg-gray-50 p-3 text-[12px]">
                  <div>topic: {semanticMap?.projectedMapTopic ?? '/semantic_map/projected_map'}</div>
                  <div>frame: {projectedMap.frameId ?? 'map'}</div>
                </div>
              </div>
            ) : (
              <p className="text-[14px] text-[#a0aec0]">No projected map received yet.</p>
            )}
          </div>

          <div className="panel">
            <h3 className="mb-3 text-[18px] text-[#2d3748]">Octomap / Legend</h3>
            <div className="space-y-3 text-[13px] text-[#2d3748]">
              <div className="rounded-lg bg-gray-50 p-3">
                <div className="text-[11px] uppercase tracking-wide text-[#a0aec0]">Octomap Cloud</div>
                <div>points: {octomapCloud?.pointCount ?? 0}</div>
                <div>topic: {semanticMap?.octomapCloudTopic ?? '/semantic_map/octomap_cloud'}</div>
                <div>frame: {octomapCloud?.frameId ?? 'map'}</div>
              </div>
              <div className="rounded-lg bg-gray-50 p-3">
                <div className="mb-2 text-[11px] uppercase tracking-wide text-[#a0aec0]">Legend</div>
                <div className="flex items-center gap-2">
                  <span className="inline-block h-3 w-3 rounded-sm bg-[#fc8181]" />
                  <span>Occupied</span>
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <span className="inline-block h-3 w-3 rounded-sm bg-[#c6f6d5]" />
                  <span>Free</span>
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <span className="inline-block h-3 w-3 rounded-sm bg-[#edf2f7]" />
                  <span>Unknown</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

