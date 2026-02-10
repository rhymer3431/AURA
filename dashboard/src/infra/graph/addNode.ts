import { NodeData } from "../../domain/graph/NodeData";

export function addNode(
    nodes: Map<string, NodeData>,
    raw: {
        id: string | number;
        label?: string;
        name?: string;
        cls?: string;
        score?: number;
        group?: number;
        // 🔥 명시적으로 위치 정보도 받을 수 있도록 추가
        x?: number;
        y?: number;
        vx?: number;
        vy?: number;
        fx?: number | null;
        fy?: number | null;
    },
) {
    const id = String(raw.id);

    // 🔥 기존 노드에서 위치 정보 보존
    const existing = nodes.get(id);

    const name =
        raw.name ?? raw.label ?? existing?.name ?? raw.cls ?? "entity";
    const group =
        existing?.group ?? raw.group ?? (name === "person" ? 1 : 2);

    // 🔥 위치 정보 우선순위:
    // 1. 기존 노드의 위치 (가장 우선)
    // 2. 새로 전달된 위치
    // 3. undefined (force simulation이 자동 배치)
    nodes.set(id, {
        ...existing,  // 기존 노드의 모든 속성 유지
        id,
        name,
        group,
        // 위치 정보: 기존 값이 있으면 무조건 유지
        x: existing?.x ?? raw.x,
        y: existing?.y ?? raw.y,
        vx: existing?.vx ?? raw.vx,
        vy: existing?.vy ?? raw.vy,
        // 고정 위치: 기존 값 우선, 없으면 새 값, 둘 다 없으면 null
        fx: existing?.fx !== undefined ? existing.fx : (raw.fx ?? null),
        fy: existing?.fy !== undefined ? existing.fy : (raw.fy ?? null),
    });
}