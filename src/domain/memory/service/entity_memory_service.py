import torch
import torch.nn.functional as F
from collections import defaultdict

class EntityMemoryService:
    def __init__(self, match_threshold=0.75, max_history=20):
        self.entities = {}      # global_id -> entity dict
        self.next_gid = 1
        self.match_threshold = match_threshold
        self.max_history = max_history

    def update_from_nodes(self, nodes):
        """
        nodes: [{track_id, cls_name, roi_feat, ...}]
        """
        updated_nodes = []

        for node in nodes:
            if node["cls_name"] != "person":
                # 사람 아닌 경우는 track_id 쓰거나 다른 메모리 구조로 처리
                updated_nodes.append(node)
                continue

            embed = node["roi_feat"].float()
            embed = F.normalize(embed, dim=0)

            gid = self._match_or_create(embed)
            node["global_id"] = gid

            # entity 업데이트
            self._update_entity(gid, embed, node)

            updated_nodes.append(node)

        return updated_nodes

    def _match_or_create(self, embed):
        best_gid = None
        best_sim = -1

        for gid, ent in self.entities.items():
            mem_embed = ent["mean_embed"]
            sim = torch.dot(embed, mem_embed).item()

            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_sim >= self.match_threshold:
            return best_gid

        # 새로운 엔티티 생성
        gid = self.next_gid
        self.next_gid += 1
        self.entities[gid] = {
            "cls_name": "person",
            "appear_embeds": [],
            "mean_embed": embed,
            "track_history": [],
        }
        return gid

    def _update_entity(self, gid, embed, node):
        ent = self.entities[gid]

        ent["appear_embeds"].append(embed)
        if len(ent["appear_embeds"]) > self.max_history:
            ent["appear_embeds"].pop(0)

        # 평균 embedding 업데이트
        ent["mean_embed"] = torch.stack(ent["appear_embeds"]).mean(dim=0)
        ent["mean_embed"] = F.normalize(ent["mean_embed"], dim=0)

        ent["track_history"].append(node["bbox"])
        self.entities[gid] = ent
