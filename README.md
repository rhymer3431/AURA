# AURA: Autonomous Understanding Reactive & Attention
YOLO-World · ByteTrack · Lightweight-GNN · SGCL · GRIN · LLM

지능형 로봇을 위한 **실시간 인지 → 추론 → 행동 결정** 전체 파이프라인을 구현한 프로젝트입니다.  
시각 기반 객체 분석, 관계 구조화, 위험 평가, 경로 예측, 고차원적 정책 추론까지 단계적으로 수행하며  
인간의 인지 시스템을 모방한 구조를 갖습니다.

---

## 1. 프로젝트 개요

본 프로젝트는 다음 여섯 가지 목표를 중심으로 설계되었습니다.

1. **실시간 객체 탐지 및 추적**  
   YOLO-World와 ByteTrack을 활용해 다중 객체를 검출 및 추적합니다.

2. **Scene Graph 기반 장면 이해**  
   Lightweight-GNN 또는 Transformer기반 SGG 모델을 통해 Scene Graph를 만듭니다.
   현재 여러 모델을 테스트 중에 있으며, 전체 파이프라인과 적합한 모델을 적용할 예정입니다.

5. **GRIN 기반 동적 이동 경로 예측**  
   정적 Scene Graph를 만든 이후, 여러 프레임동안 Scene Graph 시퀀스를 저장했다 한번에 업데이트 하는 방식으로 동적인 관계를 예측합니다.

6. **LLM Reasoning**  
   Scene Graph, 위험도, 미래 경로 정보를 종합해 행동 전략을 도출합니다.
   기존에는 Scene Graph를 템플릿에 맞춰서 LLM에 프롬프트를 전달하는 구조지만,
   최근에 VLM으로 전환하여 ROI Feature와 함께 건내주는 방식또한 고려 중입니다.

8. **Jetson Edge–Server 스트리밍 파이프라인**  
   Edge에서는 실시간 탐지·추적을, Server에서는 GNN·SGCL·GRIN·LLM을 수행하는 분산 구조입니다.

---

## 2. 파이프라인 상세 구성

### 2.1 Object Detection  
**모듈:** `infra/detector/yolo_world_adapter.py`  
- YOLO-World v2 기반  
- 오픈 도메인 객체 탐지  
- Output: `[x1, y1, x2, y2, score, class_name]`

### 2.2 Object Tracking  
**모듈:** `infra/tracker/bytetrack_adapter.py`  
- ByteTrack 다중 객체 추적  
- Output: `TrackID + Bounding Box`

### 2.3 Scene Graph Generation  
**모듈:**  
- Lightweight GNN: `lightweight_gnn.py`  
- 또는 MotifNet SGG(pretrained)  

**Scene Graph 구조:**  
- Node: 객체 ID, class, position, bbox  
- Edge: relationship label, 거리, 공간적 순서(spatial ordering)

### 2.4 SGCL  
Static Geometric & Context Logic  
- Scene Graph 기반 정적·공간 분석  
- 위험도 판단  
- 잠재적 충돌 가능성 평가

### 2.5 GRIN Trajectory Prediction  
- 동적 객체의 향후 위치 예측  
- SGCL 출력과 결합하여 Risk Score 업데이트

### 2.6 LLM Reasoning  
- Scene Graph + Risk Score + Trajectory 기반 행동 전략 생성

---

## 3. 실행 방법

### 3.1 환경 구성
```bash
uv sync
# 또는
pip install -r requirements.txt
