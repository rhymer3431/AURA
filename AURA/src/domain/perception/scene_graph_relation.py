from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RelationType(Enum):
    """
    기본적인 장면 관계 유형.
    실제 perception layer에서는 RELATION_ID_TO_NAME / RELATION_NAME_TO_ID
    맵핑과 조합하여 확장된 relation space가 사용될 수 있음.
    """
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    OVERLAPS = "overlaps"
    NEAR = "near"


@dataclass(frozen=True)
class SceneGraphRelation:
    """
    Scene Graph에서 하나의 관계를 표현하는 Domain Model.

    - subject_id: 주체 entity의 LTM entity_id (또는 SG frame 내 entity index)
    - predicate: relation ID (int)
    - object_id: 객체 entity의 LTM entity_id
    - confidence: 관계 신뢰도 (0.0 ~ 1.0)
    """

    subject_id: int
    predicate: int  # relation_id
    object_id: int
    confidence: float = 1.0

    # relation_name_map은 perception adapter에서 제공됨
    def name(self, relation_id_to_name: dict[int, str]) -> str:
        return relation_id_to_name.get(self.predicate, str(self.predicate))

    def to_tuple(self):
        """
        diff 계산 처리 시 key로 사용되는 불변 tuple.
        """
        return (self.subject_id, self.predicate, self.object_id)

    def __hash__(self):
        """
        관계의 identity는 (subject, predicate, object).

        confidence는 identity를 변화시키지 않음 → diff 업데이트에서 유리.
        """
        return hash((self.subject_id, self.predicate, self.object_id))

    def __eq__(self, other):
        if not isinstance(other, SceneGraphRelation):
            return False
        return (
            self.subject_id == other.subject_id
            and self.predicate == other.predicate
            and self.object_id == other.object_id
        )


class SceneGraphRelations:
    """
    관계 리스트를 다루기 위한 간단한 Domain Collection.

    SceneGraphFrame.relations 는 단순 list[int,int,int] 형태라서
    domain 모델 기반 편의 기능을 제공하는 wrapper.
    """

    def __init__(self, relations: list[SceneGraphRelation]):
        self._relations = relations

    def __iter__(self):
        return iter(self._relations)

    def __len__(self):
        return len(self._relations)

    def to_raw_tuples(self):
        """
        SGFrame 저장용 (subject_idx, object_idx, relation_id) 형태로 변환.
        """
        return [
            (rel.subject_id, rel.object_id, rel.predicate)
            for rel in self._relations
        ]

    def filter_by_subject(self, entity_id: int) -> list[SceneGraphRelation]:
        return [r for r in self._relations if r.subject_id == entity_id]

    def filter_by_object(self, entity_id: int) -> list[SceneGraphRelation]:
        return [r for r in self._relations if r.object_id == entity_id]

    def filter_by_predicate(self, predicate: int) -> list[SceneGraphRelation]:
        return [r for r in self._relations if r.predicate == predicate]
