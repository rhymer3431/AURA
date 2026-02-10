import { LinkData } from "../../domain/graph/LinkData";
import { NodeData } from "../../domain/graph/NodeData";
import { Triplet } from "../../domain/graph/Triplet";
import { addEdge } from "./addEdge";

import { addNode } from "./addNode";
import { findLabelForEntityId } from "./findLabelForEntityId";

export type SceneGraphEdgePayload = {
    subject?: number | string | NodeData;
    object?: number | string | NodeData;
    predicate?: string;
    subject_id?: number | string | NodeData;
    object_id?: number | string | NodeData;
    relation?: string;
    confidence?: number;
    type?: "static" | "temporal";
};

function resolveId(raw: unknown): string | undefined {
    if (typeof raw === "string" || typeof raw === "number") {
        return String(raw);
    }
    if (raw && typeof raw === "object" && "id" in (raw as any)) {
        const value = (raw as any).id;
        if (typeof value === "string" || typeof value === "number") {
            return String(value);
        }
    }
    return undefined;
}

export function upsertEdgeFromPayload(
    edge: SceneGraphEdgePayload,
    nodes: Map<string, NodeData>,
    links: Map<string, LinkData>,
    triplets: Triplet[],
    meta: any,
): Triplet[] {
    const subjectId = resolveId(edge.subject_id ?? edge.subject);
    const objectId = resolveId(edge.object_id ?? edge.object);

    if (!subjectId || !objectId) {
        return triplets;
    }

    const predicate = edge.predicate ?? edge.relation ?? "related to";
    const type: Triplet["type"] =
        edge.type === "temporal" ? "temporal" : "static";
    const confidence =
        typeof edge.confidence === "number" ? edge.confidence : 0.75;

    const subjectName =
        nodes.get(subjectId)?.name ?? findLabelForEntityId(meta, subjectId);
    const objectName =
        nodes.get(objectId)?.name ?? findLabelForEntityId(meta, objectId);

    addNode(nodes, { id: subjectId, name: subjectName });
    addNode(nodes, { id: objectId, name: objectName });

    addEdge(links, {
        subject: subjectId,
        predicate,
        object: objectId,
        confidence,
        type,
    });

    const idx = triplets.findIndex(
        (t) =>
            String(t.subject_id) === subjectId &&
            String(t.object_id) === objectId &&
            t.predicate === predicate,
    );

    const nextTriplet: Triplet = {
        subject: subjectName ?? "entity",
        subject_id: (edge.subject_id ?? edge.subject ?? subjectId) as string | number,
        predicate,
        object: objectName ?? "entity",
        object_id: (edge.object_id ?? edge.object ?? objectId) as string | number,
        confidence,
        type,
    };

    if (idx >= 0) {
        const clone = [...triplets];
        clone[idx] = nextTriplet;
        return clone;
    }

    return [...triplets, nextTriplet];
}
