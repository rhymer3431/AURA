import { GraphData } from "./GraphData";
import { NodeData } from "./NodeData";

export interface ForcedGraphProps {
    data: GraphData;
    width?: number;
    height?: number;
    nodeSize?: number;
    onNodeClick?: (node: NodeData) => void;
    colorFunction?: (node: NodeData) => string;
}

export type { GraphData } from "./GraphData";
export type { NodeData } from "./NodeData";
