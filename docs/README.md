# AURA Architecture Docs

이 디렉토리는 현재 `main` 기준의 AURA 런타임 구조를 설명한다.

기준 범위:

- G1 runtime 중앙 서버 전환
- A-F 단계까지 반영된 write/read model
- worker contract, observability, recovery state machine

문서 구성:

- [architecture/AURA_RUNTIME_ARCHITECTURE.md](./architecture/AURA_RUNTIME_ARCHITECTURE.md)
  - 현재 중앙 서버 구조와 G1 실행 흐름
- [architecture/STATE_AND_OBSERVABILITY.md](./architecture/STATE_AND_OBSERVABILITY.md)
  - write-side state, read-side snapshot, dashboard/WebRTC/legacy mirror
- [architecture/WORKER_CONTRACTS_AND_RECOVERY.md](./architecture/WORKER_CONTRACTS_AND_RECOVERY.md)
  - worker request/result contract, validation, recovery state machine
- [architecture/MIGRATION_SUMMARY_A_TO_F.md](./architecture/MIGRATION_SUMMARY_A_TO_F.md)
  - A-F 단계별 변경 요약

문서 원칙:

- 현재 코드를 기준으로 서술한다.
- 과거 bridge/session/orchestrator 중심 설명은 더 이상 canonical하지 않다.
- G/H 범위는 아직 완료되지 않았으므로 이 문서들은 A-F 완료 상태까지만 확정적으로 다룬다.
