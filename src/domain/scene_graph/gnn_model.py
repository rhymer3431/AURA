# src/robotics/domain/scene_graph/gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, h, adj):
        agg = torch.matmul(adj, h)
        deg = adj.sum(1, keepdim=True) + 1e-6
        agg = agg / deg
        out = torch.cat([h, agg], dim=1)
        return F.relu(self.linear(out))


class LightweightGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_relations):
        super().__init__()
        self.gnn1 = GraphSAGELayer(node_dim, hidden_dim)
        self.gnn2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_relations)
        )
        self.num_relations = num_relations

    def forward(self, node_feat, edge_feat, adj):
        h = self.gnn1(node_feat, adj)
        h = self.gnn2(h, adj)

        N = h.size(0)
        logits = torch.zeros(N, N, self.num_relations)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                x = torch.cat([h[i], h[j], edge_feat[i][j]], dim=-1)
                logits[i][j] = self.classifier(x)

        return logits
