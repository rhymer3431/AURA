import type { StreamEntity } from "../streaming/streamTypes";

export type FrameSize = { width: number; height: number };

export type BoundingBoxOverlay = {
  id: string;
  label: string;
  confidence?: number;
  color: string;
  box: {
    leftPct: number;
    topPct: number;
    widthPct: number;
    heightPct: number;
  };
};

const COLOR_PALETTE = [
  "#4fd1c5",
  "#22c55e",
  "#3b82f6",
  "#f59e0b",
  "#ef4444",
  "#a855f7",
  "#06b6d4",
  "#ec4899",
];

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

const pickColor = (entity: StreamEntity, idx: number) => {
  const key = entity.trackId ?? entity.entityId ?? idx;
  const paletteIdx = Math.abs(key) % COLOR_PALETTE.length;
  return COLOR_PALETTE[paletteIdx];
};

const normalizeBox = (box: number[], frameSize?: FrameSize) => {
  const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = box;
  const isPixelSpace =
    frameSize && Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) > 1;

  const nx1 = isPixelSpace && frameSize ? x1 / frameSize.width : x1;
  const ny1 = isPixelSpace && frameSize ? y1 / frameSize.height : y1;
  const nx2 = isPixelSpace && frameSize ? x2 / frameSize.width : x2;
  const ny2 = isPixelSpace && frameSize ? y2 / frameSize.height : y2;

  const left = clamp01(Math.min(nx1, nx2));
  const right = clamp01(Math.max(nx1, nx2));
  const top = clamp01(Math.min(ny1, ny2));
  const bottom = clamp01(Math.max(ny1, ny2));

  return {
    leftPct: left,
    topPct: top,
    widthPct: clamp01(right - left),
    heightPct: clamp01(bottom - top),
  };
};

export function buildBoundingBoxOverlays(
  entities: StreamEntity[],
  frameSize?: FrameSize,
): BoundingBoxOverlay[] {
  return entities
    .filter((entity) => Array.isArray(entity.box) && entity.box.length >= 4)
    .map((entity, idx) => ({
      id: `bbox-${entity.entityId ?? idx}`,
      label: `${entity.cls ?? "Entity"} #${entity.entityId ?? entity.trackId ?? idx}`,
      confidence: typeof entity.score === "number" ? entity.score : undefined,
      color: pickColor(entity, idx),
      box: normalizeBox(entity.box, frameSize),
    }))
    .filter((overlay) => overlay.box.widthPct > 0 && overlay.box.heightPct > 0);
}
// --- Improved Purity-UI Style BBox Overlay ---
export const overlayStyles: React.CSSProperties = {
  position: "absolute",
  border: "2.2px solid currentColor",
  borderRadius: "10px",
  boxSizing: "border-box",
  background: "transparent",
  padding: "4px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  transition: "border-color 0.2s ease-out, box-shadow 0.2s ease-out",

  // Hover-like 효과 (시각적 안정감)
  boxShadow: "0 0 0 1px rgba(0,0,0,0.06)",
};

// --- Centered Minimal Label (Glass-like) ---
export const labelStyles: React.CSSProperties = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  fontSize: "13px",
  fontWeight: 500,
  color: "#2d3748",
  textAlign: "center",
  padding: "3px 6px",
  whiteSpace: "nowrap",
  userSelect: "none",
  pointerEvents: "none",

  // ✨ Semi-glass effect
  background: "rgba(255, 255, 255, 0.45)",
  backdropFilter: "blur(6px)",
  borderRadius: "8px",

  // 텍스트 가독성 향상
  textShadow: "0 1px 2px rgba(0, 0, 0, 0.12)",
};
