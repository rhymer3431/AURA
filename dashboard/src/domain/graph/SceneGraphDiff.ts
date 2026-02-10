import { LinkData } from "./LinkData";
import { NodeData } from "./NodeData";

export type SceneGraphDiff = {
    nodesAdded?: NodeData[];
    nodesRemoved?: Array<string | number>;
    nodesUpdated?: NodeData[];
    edgesAdded?: LinkData[];
    edgesRemoved?: Array<
        | LinkData
        | { source: string | number | NodeData; target: string | number | NodeData; predicate?: string }
        | string
    >;
};
