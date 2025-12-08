# AURA: Autonomous Understanding Reactive & Attention  
**YOLO-World · ByteTrack · Scene Graph Generation · SGCL · GRIN · LLM Reasoning**

AURA는 실시간 로봇 인지 시스템을 구축하기 위한 프로젝트로서,  
카메라 입력으로부터 객체 탐지, 추적, 장면 구조화, 위험도 분석, 경로 예측, 정책 추론까지  
전체 인지 파이프라인을 통합하여 동작하도록 설계되었습니다.  

본 시스템은 Jetson Edge 장치에서의 경량 연산과 서버에서의 고성능 추론을 조합하는  
Edge–Server 분산 구조를 기반으로 합니다.

---

## 1. 시스템 개요

본 프로젝트는 다음 기능을 중심으로 구성됩니다.

### 1) 실시간 객체 탐지 및 추적
- YOLO-World 기반 오픈 도메인 객체 탐지를 수행합니다.  
- ByteTrack을 사용하여 다중 객체를 추적합니다.  
- Edge 환경에서 실시간 성능을 확보하도록 최적화되어 있습니다.

### 2) Scene Graph 기반 장면 구조화
- Lightweight-GNN 또는 Transformer 기반 Scene Graph Generation(SGG) 모델을 사용합니다.  
- 객체 간의 공간적·관계적 정보를 구조화하여 후속 단계의 입력으로 제공합니다.

### 3) SGCL (Static Geometric & Context Logic)
- Scene Graph를 기반으로 정적 기하 분석을 수행합니다.  
- 거리, 방향, 문맥 기반 규칙을 통해 위험도를 평가합니다.

### 4) GRIN 기반 동적 경로 예측
- Scene Graph 시퀀스를 입력으로 GRIN(Gaussian Recurrent Interaction Network) 모델을 적용합니다.  
- 객체의 미래 위치를 예측하고 잠재적 충돌 가능성을 산출합니다.  
- SGCL 결과와 결합하여 동적 위험도를 계산합니다.

### 5) LLM Reasoning
- Scene Graph, 위험도 분석 결과, 경로 예측 정보를 입력으로 정책을 생성합니다.  
- 상황 요약, 주의 대상(focus targets), 행동 전략 등을 산출합니다.  
- ROI Feature 기반 VLM 입력 방식도 실험 중입니다.

### 6) Jetson Edge–Server 파이프라인
- Edge에서는 탐지·추적을 수행합니다.  
- Server에서는 Scene Graph 생성, SGCL, GRIN, LLM Reasoning을 수행합니다.  
- WebRTC 기반으로 실시간 영상 및 metadata를 송수신합니다.

---

## 2. 파이프라인 구성

### 2.1 Object Detection  
**파일:** `infra/detector/yolo_world_adapter.py`  
- YOLO-World v2 기반 탐지를 수행합니다.  
- 출력 형식은 다음과 같습니다.
[x1, y1, x2, y2, score, class_name]

yaml
코드 복사

---

### 2.2 Object Tracking  
**파일:** `infra/tracker/bytetrack_adapter.py`  
- ByteTrack 기반 다중 객체 추적을 수행합니다.  
- `track_id + bounding box` 정보를 출력합니다.

---

### 2.3 Scene Graph Generation  
**파일:** `lightweight_gnn.py` 또는 Transformer 기반 SGG  
- Node 정보: 객체 ID, class, bbox, 위치  
- Edge 정보: 관계(label), 거리, 방향(spatial ordering)  
- 생성된 Scene Graph는 GRIN 및 SGCL의 입력으로 사용됩니다.

---

### 2.4 SGCL  
정적 기하 분석 및 문맥 기반 규칙을 적용하여  
객체 간 충돌 가능성, 위험도, 비정상 패턴 등을 판단합니다.

---

### 2.5 GRIN Trajectory Prediction  
- Scene Graph 시퀀스를 기반으로 객체의 미래 경로를 예측합니다.  
- 잠재적 충돌 위치 및 동적 위험도를 산출합니다.  
- SGCL 결과와 통합하여 종합 Risk Score를 계산합니다.

---

### 2.6 LLM Reasoning  
- Scene Graph, Risk Score, trajectory 정보를 입력으로 정책을 생성합니다.  
- 상황 설명, 행동 전략, 탐지 우선순위 등을 출력합니다.  

---

## 3. 메모리 구조(STM / LTM)

AURA는 객체의 지속성과 의미적 안정성을 유지하기 위해  
단기 메모리(STM)와 장기 메모리(LTM)를 구분하여 사용합니다.

### 3.1 Short-Term Memory (STM)
- 최근 프레임 기반의 임시 상태를 저장합니다.  
- Track History, 속도 벡터, 최근 Scene Graph window 등을 포함합니다.  
- 유지 기간은 수 프레임 수준이며 GRIN 및 추적 안정화에 사용됩니다.

### 3.2 Long-Term Memory (LTM)
- 장기간 유지되는 객체의 정체성과 의미적 패턴을 저장합니다.  
- Appearance embedding, Scene Graph 관계 이력, 장기 위험 패턴 등을 포함합니다.  
- 유지 기간은 수십 초 이상이며 Re-ID 및 LLM Reasoning에서 사용됩니다.

---

## 4. Jetson Edge–Server 구조

### Edge (Jetson)
- YOLO-World 기반 탐지와 ByteTrack 추적을 수행합니다.  
- WebRTC를 통해 서버로 영상 및 metadata를 전송합니다.

### Server
- Scene Graph 생성, SGCL, GRIN, LLM Reasoning을 수행합니다.  
- React 기반 인터페이스를 통해 실시간으로 결과를 시각화합니다.

---

## 5. 실행 방법

### 5.1 환경 구성

#### uv 기반 설치
```bash
uv sync
