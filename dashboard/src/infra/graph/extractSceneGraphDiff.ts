import { SceneGraphDiff } from "../../domain/graph/SceneGraphDiff";

export function extractSceneGraphDiff(meta: any): SceneGraphDiff | null {
    if (!meta) return null;

    const direct = meta.sceneGraphDiff ?? meta.scene_graph_diff;
    if (direct) return direct as SceneGraphDiff;

    const hasAny =
        "nodesAdded" in meta ||
        "nodes_removed" in meta ||
        "nodesRemoved" in meta ||
        "edgesAdded" in meta ||
        "edgesRemoved" in meta;

    if (hasAny) {
        return {
            nodesAdded: meta.nodesAdded ?? [],
            nodesRemoved: meta.nodesRemoved ?? meta.nodes_removed ?? [],
            edgesAdded: meta.edgesAdded ?? [],
            edgesRemoved: meta.edgesRemoved ?? [],
        };
    }

    return null;
}
