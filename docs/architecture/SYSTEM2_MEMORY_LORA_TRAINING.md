# System2 Memory LoRA Training

`isaac-aura`의 System 2는 대화 응답기가 아니라 `pixel_goal / yaw / stop / wait` 문자열을 내는 memory-aware navigation planner다. 그래서 이 학습 셋업은 chat/router를 섞지 않고, `MemoryContextBundle -> System2Session` 계약을 그대로 재현하는 LoRA SFT 경로만 다룬다.

## Added Files

- `src/inference/training/system2_memory_lora.py`
  - synthetic seed dataset builder
  - dataset validator
  - JSONL -> multimodal message renderer
- `src/inference/training/system2_memory_lora_train.py`
  - HF/PEFT 기반 LoRA trainer
  - `--validate-only` 지원
- `experiments/system2_memory_lora/environment.windows.yml`
  - Windows + Conda 기준 GPU 학습 환경
- `experiments/system2_memory_lora/system2_memory_lora.yaml`
  - 기본 학습 하이퍼파라미터
- `scripts/powershell/build_system2_memory_lora_dataset.ps1`
- `scripts/powershell/setup_system2_memory_lora_env.ps1`
- `scripts/powershell/run_system2_memory_lora_training.ps1`

## Dataset Schema

각 샘플은 `train.jsonl`, `val.jsonl`, `test.jsonl`에 한 줄씩 저장된다.

- `instruction`: 한국어 자연어 명령
- `decision_text`: 기대하는 System 2 출력 (`<y>, <x>`, `STOP`, `←`, `→`, `↓`)
- `decision_mode`: `pixel_goal`, `stop`, `yaw_left`, `yaw_right`, `wait`
- `current_image`: 현재 프레임 이미지 경로
- `history_images`: 최근 히스토리 프레임 목록
- `memory_context`: `MemoryContextBundle` 직렬화 결과
- `events`: `force_s2`, `stuck`, `collision_risk`, `task_state`

Seed dataset은 다음 task family를 고르게 포함한다.

- `direct_pixel`
- `memory_pixel`
- `memory_turn_left`
- `memory_turn_right`
- `stop`
- `wait`

## Build Dataset

```powershell
.\scripts\powershell\build_system2_memory_lora_dataset.ps1
```

기본 출력 경로는 `artifacts/datasets/system2_memory_lora_seed`다.

## Setup Environment

```powershell
.\scripts\powershell\setup_system2_memory_lora_env.ps1
```

이 스크립트는 `aura-system2-lora` Conda 환경을 만들고, repo를 editable install 한다. 현재 기본 Python 환경의 `torch`가 깨져 있어도 별도 환경으로 격리된다.

## Validate Training Inputs

```powershell
.\scripts\powershell\run_system2_memory_lora_training.ps1 -ValidateOnly
```

이 모드는 dataset 경로, split 수, config 해석 결과만 검증한다.

## Run Training

`experiments/system2_memory_lora/system2_memory_lora.yaml`의 `model_name_or_path`를 HF 형식의 InternVLA System 2 체크포인트 경로로 바꾼 뒤 실행한다.

```powershell
.\scripts\powershell\run_system2_memory_lora_training.ps1 -ModelNameOrPath C:\models\InternVLA-N1-System2-HF
```

## Notes

- 현재 학습 스크립트는 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`를 우선 LoRA 대상으로 삼는다.
- `multi_modal_projector`, `lm_head`는 존재할 때만 `modules_to_save`로 유지한다.
- seed dataset은 smoke test와 prompt-format stabilization 용도다. 실제 성능을 내려면 Isaac Sim 또는 실로봇 로그를 같은 schema로 추가해야 한다.
