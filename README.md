# Robotics Perception & Reasoning Pipeline  
YOLO-World · ByteTrack · Lightweight-GNN · SGCL · GRIN · LLM

지능형 로봇을 위한 **실시간 인지 → 추론 → 행동 결정** 전체 파이프라인을 구현한 프로젝트입니다.  
시각 기반 객체 분석, 관계 구조화, 위험 평가, 경로 예측, 고차원적 정책 추론까지 단계적으로 수행하며  
인간의 인지 시스템을 모방한 구조를 갖습니다.

---

## 1. 프로젝트 개요

본 프로젝트는 다음 여섯 가지 목표를 중심으로 설계되었습니다.

1. **실시간 객체 탐지·추적**  
   YOLO-World(Open-Vocabulary Detection)과 ByteTrack을 활용해 다중 객체를 검출·추적합니다.

2. **Scene Graph 기반 장면 이해**  
   Lightweight-GNN 또는 MotifNet(SGG)을 통해 노드(객체)와 엣지(관계)를 구조화합니다.

3. **SGCL 기반 정적·공간 논리 추론**  
   Scene Graph로부터 거리, 방향, 상호작용 가능성 등 공간적 맥락을 분석합니다.

4. **GRIN 기반 동적 이동 경로 예측**  
   움직이는 객체의 향후 움직임을 예측해 위험도를 보정합니다.

5. **LLM Reasoning**  
   Scene Graph, 위험도, 미래 경로 정보를 종합해 행동 전략을 도출합니다.

6. **Jetson Edge–Server 스트리밍 파이프라인**  
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
- 예: “전방 2m 내 충돌 위험. 속도를 0.8m/s로 감소하세요.”

---

## 3. 실행 방법

### 3.1 환경 구성
```bash
uv sync
# 또는
pip install -r requirements.txt
