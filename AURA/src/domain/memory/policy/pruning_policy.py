# src/domain/memory/pruning_policy.py
from src.domain.memory.entity_record import EntityRecord

class PruningPolicy:
    """
    엔티티의 삭제 여부를 판단하는 정책 객체.
    프레임 시퀀스에서 불안정하거나 오래된 엔티티를 제거.
    """

    def should_delete(
        self,
        entity: EntityRecord,
        frame_idx: int,
    ) -> (bool | str):
        """
        Returns:
            (삭제 여부, 이유 문자열)
        """
        raise NotImplementedError
