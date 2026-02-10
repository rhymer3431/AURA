import { GraphData } from "./GraphData";
import { NodeData } from "./NodeData";

export type GraphController = {
    update: (
        nextData: GraphData,
        opts?: {
            width?: number;
            height?: number;
            nodeSize?: number;
            onNodeClick?: (node: NodeData) => void;
            colorFunction?: (node: NodeData) => string;
        }
    ) => void;
    stop: () => void;
};