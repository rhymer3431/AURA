export function addEdge(
    links: Map<string, any>,
    edge: {
        subject: string | number;
        predicate: string;
        object: string | number;
        confidence?: number;
        type?: "static" | "temporal";
    }
) {
    const subjectId = String(edge.subject);
    const objectId = String(edge.object);
    const predicate = edge.predicate;

    const key = `${subjectId}:${predicate}:${objectId}`;

    links.set(key, {
        source: subjectId,
        target: objectId,
        predicate,
        confidence: edge.confidence ?? 1.0,
        type: edge.type ?? "static",
    });
}

// 🔑 Edge key 포맷은 서버의 _serialize_edge_key 와 맞추어 주세요.
export function makeEdgeKey(subjectId: string, predicate: string, objectId: string): string {
    return `${subjectId}:${predicate}:${objectId}`;
}
