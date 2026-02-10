import { LinkData } from "./LinkData";
import { NodeData } from "./NodeData";

export interface GraphData {
    nodes: NodeData[];
    links: LinkData[];
}