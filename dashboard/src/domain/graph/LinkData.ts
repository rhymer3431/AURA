import { NodeData } from "./NodeData";

export type LinkData = {
    id?: string | number;
    source: string | NodeData;
    target: string | NodeData;
    predicate?: string;
    type?: "static" | "temporal";
    confidence?: number;
};
