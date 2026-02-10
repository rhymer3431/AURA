import { InternalGraphState } from "../../domain/graph/InternalGraphState";
import { NodeData } from "../../domain/graph/NodeData";
import { SceneGraphDiff } from "../../domain/graph/SceneGraphDiff";
import { makeEdgeKey } from "./addEdge";
import { addNode } from "./addNode";
import { findLabelForEntityId } from "./findLabelForEntityId";
import { SceneGraphEdgePayload, upsertEdgeFromPayload } from "./upsertEdgeFromPayload";

const resolveId = (raw: unknown): string | undefined => {
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
};

const extractPositionData = (node: NodeData) => ({
    x: node.x,
    y: node.y,
    vx: node.vx,
    vy: node.vy,
    fx: node.fx,
    fy: node.fy,
});

const mergeNodeWithPosition = (
    existingNode: NodeData | undefined,
    newNodeData: Partial<NodeData>
): Partial<NodeData> => {
    const source = existingNode ?? {};
    return {
        ...newNodeData,
        x: newNodeData.x ?? source.x,
        y: newNodeData.y ?? source.y,
        vx: newNodeData.vx ?? source.vx,
        vy: newNodeData.vy ?? source.vy,
        fx: newNodeData.fx ?? source.fx,
        fy: newNodeData.fy ?? source.fy,
    };
};

const attachPositionHint = (
    id: string,
    data: Partial<NodeData>,
    positionHints?: Map<string, Partial<NodeData>>
) => {
    const hint = positionHints?.get(id);
    if (!hint) return data;
    return mergeNodeWithPosition(undefined, { ...data, ...hint });
};

export function applySceneGraphDiff(
    prev: InternalGraphState,
    diff: SceneGraphDiff,
    meta: any,
    positionHints?: Map<string, Partial<NodeData>>,
): InternalGraphState {
    const nodes = new Map(prev.nodes);
    const links = new Map(prev.links);
    let triplets = [...prev.triplets];

    diff.nodesAdded?.forEach((n) => {
        const id = resolveId((n as any).id);
        if (!id) return;

        const existingNode = nodes.get(id);
        const newNodeData: NodeData = {
            id,
            name: (n as any).name ?? (n as any).label ?? findLabelForEntityId(meta, id),
            group: (n as any).group,
            ...(existingNode ? extractPositionData(existingNode) : {}),
        };

        const merged = mergeNodeWithPosition(
            existingNode,
            attachPositionHint(id, newNodeData, positionHints),
        );

        addNode(nodes, merged as NodeData);
    });

    diff.nodesUpdated?.forEach((n) => {
        const id = resolveId((n as any).id);
        if (!id) return;

        const existingNode = nodes.get(id);
        const updatedData = {
            id,
            name: (n as any).name ?? (n as any).label ?? findLabelForEntityId(meta, id),
            group: (n as any).group,
        };

        const mergedNode = mergeNodeWithPosition(
            existingNode,
            attachPositionHint(id, updatedData, positionHints),
        );
        addNode(nodes, mergedNode as NodeData);
    });

    diff.edgesAdded?.forEach((edge) => {
        triplets = upsertEdgeFromPayload(
            edge as SceneGraphEdgePayload,
            nodes,
            links,
            triplets,
            meta,
        );
    });

    diff.edgesRemoved?.forEach((edge) => {
        let key: string | null = null;

        if (typeof edge === "string") {
            key = edge;
        } else {
            const subjectIdRaw = (edge as any).subject_id ?? (edge as any).subject ?? (edge as any).source;
            const objectIdRaw = (edge as any).object_id ?? (edge as any).object ?? (edge as any).target;
            const predicate = (edge as any).predicate ?? (edge as any).relation;

            const subjectId = resolveId(subjectIdRaw);
            const objectId = resolveId(objectIdRaw);

            if (subjectId && objectId && predicate) {
                key = makeEdgeKey(subjectId, String(predicate), objectId);
            }
        }

        if (!key) return;

        const existing = links.get(key);
        if (existing) {
            const subjectId =
                typeof existing.source === "string"
                    ? existing.source
                    : (existing.source as NodeData).id;
            const objectId =
                typeof existing.target === "string"
                    ? existing.target
                    : (existing.target as NodeData).id;
            const predicate = existing.predicate ?? "";

            links.delete(key);

            triplets = triplets.filter(
                (t) =>
                    !(
                        String(t.subject_id) === subjectId &&
                        String(t.object_id) === objectId &&
                        t.predicate === predicate
                    ),
            );
        } else {
            const parts = key.split(":");
            if (parts.length === 3) {
                const [sId, pred, oId] = parts;
                triplets = triplets.filter(
                    (t) =>
                        !(
                            String(t.subject_id) === sId &&
                            String(t.object_id) === oId &&
                            t.predicate === pred
                        ),
                );
            }
        }
    });

    diff.nodesRemoved?.forEach((idRaw) => {
        const id = resolveId(idRaw);
        if (!id || !nodes.has(id)) return;

        nodes.delete(id);

        Array.from(links.keys()).forEach((key) => {
            const link = links.get(key);
            if (!link) return;
            const sId =
                typeof link.source === "string"
                    ? link.source
                    : (link.source as NodeData).id;
            const oId =
                typeof link.target === "string"
                    ? link.target
                    : (link.target as NodeData).id;

            if (sId === id || oId === id) {
                links.delete(key);
            }
        });

        triplets = triplets.filter(
            (t) =>
                String(t.subject_id) !== id &&
                String(t.object_id) !== id,
        );
    });

    return { nodes, links, triplets };
}
