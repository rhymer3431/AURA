import { LinkData } from "../domain/graph/LinkData";
import { NodeData } from "../domain/graph/NodeData";

const BASE_RADIUS = 18;
const APPROX_CHAR_WIDTH = 6;
const TEXT_PADDING = 6;
export const getRadius = (d: NodeData, nodeSize: number) => {
    const label = getLabel(d);
    const approxTextWidth = label.length * APPROX_CHAR_WIDTH;
    const textBasedRadius = approxTextWidth / 2 + TEXT_PADDING;
    return Math.max(BASE_RADIUS, nodeSize, textBasedRadius);
};

export const getLinkKey = (d: LinkData) => {
    const id = (d as any).id ?? (d as any).relationId;
    if (id !== undefined && id !== null) return String(id);
    const sourceId = typeof d.source === "string" ? d.source : String((d.source as NodeData).id);
    const targetId = typeof d.target === "string" ? d.target : String((d.target as NodeData).id);
    return `${sourceId}->${targetId}::${d.predicate ?? ""}`;
};

export const getLabel = (d: NodeData) => d.name ?? String(d.id);

