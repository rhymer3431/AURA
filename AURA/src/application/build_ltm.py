# src/application/build_ltm.py
from src.infrastructure.memory.feature_similarity_adapter import FeatureSimilarityAdapter
from src.domain.memory.entity_long_term_memory import EntityLongTermMemory
from src.domain.memory.policy.default_pruning_policy import DefaultPruningPolicy

def build_ltm(device="cuda"):
    similarity = FeatureSimilarityAdapter()
    prune_policy = DefaultPruningPolicy()
    return EntityLongTermMemory(
        similarity=similarity,
        prune_policy=prune_policy,
    )
