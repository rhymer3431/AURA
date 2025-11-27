# scene_graph_viz.py
import matplotlib.pyplot as plt
import networkx as nx


def build_scene_graph(objects, relations):
    """
    objects: [{"id": int, "label": str}, ...]
    relations: [{"subj": int, "pred": str, "obj": int}, ...]
    """
    G = nx.DiGraph()

    # 1) 노드 추가
    for obj in objects:
        node_id = obj["id"]
        label = obj.get("label", str(node_id))
        G.add_node(node_id, label=label)

    # 2) 엣지 추가 (관계)
    for rel in relations:
        s = rel["subj"]
        o = rel["obj"]
        p = rel["pred"]
        G.add_edge(s, o, label=p)

    return G


def visualize_scene_graph(G, figsize=(6, 4)):
    plt.figure(figsize=figsize)

    # 노드 위치 (spring layout: force-directed)
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # 노드/엣지/라벨 분리해서 그리기
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}

    # 노드
    nx.draw_networkx_nodes(
        G, pos,
        node_size=800,
        node_color="#EEEEFF",
        edgecolors="#333333",
        linewidths=1.5,
    )

    # 노드 라벨
    nx.draw_networkx_labels(
        G, pos, labels=node_labels,
        font_size=9,
        font_weight="bold"
    )

    # 엣지
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="->",
        arrowsize=15,
        width=1.2,
        connectionstyle="arc3,rad=0.1"
    )

    # 엣지(관계) 라벨
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 예시 데이터
    objects = [
        {"id": 1, "label": "person"},
        {"id": 2, "label": "dog"},
        {"id": 3, "label": "ball"},
    ]

    relations = [
        {"subj": 1, "pred": "holding", "obj": 3},
        {"subj": 2, "pred": "looking at", "obj": 1},
    ]

    G = build_scene_graph(objects, relations)
    visualize_scene_graph(G)
