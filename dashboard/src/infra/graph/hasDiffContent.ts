import { SceneGraphDiff } from "../../domain/graph/SceneGraphDiff";

export function hasDiffContent(diff: SceneGraphDiff | null): diff is SceneGraphDiff {
    if (!diff) return false;
    return Boolean(
        (diff.nodesAdded && diff.nodesAdded.length) ||
        (diff.nodesRemoved && diff.nodesRemoved.length) ||
        (diff.edgesAdded && diff.edgesAdded.length) ||
        (diff.edgesRemoved && diff.edgesRemoved.length),
    );
}
