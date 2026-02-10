

import { LinkData } from "./LinkData";
import { NodeData } from "./NodeData";
import { Triplet } from "./Triplet";

export type InternalGraphState = {
    nodes: Map<string, NodeData>;
    links: Map<string, LinkData>;
    triplets: Triplet[];
};