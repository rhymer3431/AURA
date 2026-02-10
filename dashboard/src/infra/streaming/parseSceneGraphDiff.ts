import type {
    BackendEdgeKey,
    BackendRelation,
    RawSceneGraphDiff,
    StreamEntity,
} from "../../domain/streaming/streamTypes";

/**
 * Transport-layer SG diff parser.
 * - Validates backend payload
 * - Ensures each field has correct array type
 * - DOES NOT convert to domain-layer types
 */
export function parseSceneGraphDiff(
    raw: unknown
): RawSceneGraphDiff | null {
    if (!raw || typeof raw !== "object") {
        return null;
    }

    const msg = raw as any;

    // ---- nodesAdded: StreamEntity[] ----
    const nodesAdded: StreamEntity[] | undefined = Array.isArray(
        msg.nodesAdded
    )
        ? msg.nodesAdded.filter(
            (n: any) => n && typeof n === "object"
        )
        : undefined;

    // ---- nodesRemoved: (number|string)[] ----
    const nodesRemoved: Array<number | string> | undefined =
        Array.isArray(msg.nodesRemoved)
            ? msg.nodesRemoved.filter(
                (id: any) =>
                    typeof id === "number" || typeof id === "string"
            )
            : undefined;

    // ---- edgesAdded: BackendRelation[] ----
    const parseBackendRelation = (
        e: any
    ): BackendRelation | null => {
        if (!e || typeof e !== "object") return null;

        const s = e.subject;
        const o = e.object;

        return {
            subject: s,
            predicate:
                typeof e.predicate === "string" ? e.predicate : "",
            object: o,
            confidence:
                typeof e.confidence === "number" ? e.confidence : 1.0,
            type:
                e.type === "static" || e.type === "temporal"
                    ? e.type
                    : undefined,
        };
    };

    const edgesAdded = Array.isArray(msg.edgesAdded)
        ? msg.edgesAdded
            .map(parseBackendRelation)
            .filter(Boolean) as BackendRelation[]
        : undefined;

    // ---- edgesRemoved: BackendEdgeKey[] ----
    const parseBackendEdgeKey = (
        e: any
    ): BackendEdgeKey | null => {
        if (!e || typeof e !== "object") return null;

        const s = e.subject;
        const o = e.object;


        return {
            subject: s,
            predicate:
                typeof e.predicate === "string" ? e.predicate : "",
            object: o,
        };
    };

    const edgesRemoved = Array.isArray(msg.edgesRemoved)
        ? msg.edgesRemoved
            .map(parseBackendEdgeKey)
            .filter(Boolean) as BackendEdgeKey[]
        : undefined;

    return {
        nodesAdded,
        nodesRemoved,
        edgesAdded,
        edgesRemoved,
    };
}
