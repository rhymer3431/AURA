
export type NodeData = {
    id: string;
    name?: string;
    group?: number;
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
    fx?: number | null;
    fy?: number | null;
};

export type LinkData = {
    source: string | NodeData;
    target: string | NodeData;
    predicate?: string;
    type?: "static" | "temporal";
    confidence?: number;
};

export interface GraphData {
    nodes: NodeData[];
    links: LinkData[];
}

interface ForcedGraphProps {
    data: GraphData;
    width?: number;
    height?: number;
    nodeSize?: number;
    onNodeClick?: (node: NodeData) => void;
    colorFunction?: (node: NodeData) => string;
}
