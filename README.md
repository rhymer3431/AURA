# AURA: Autonomous Understanding Reactive & Attention  
**YOLO-World · ByteTrack · Scene Graph Generation · SGCL · GRIN · LLM Reasoning**

AURA는 실시간 로봇 인지 시스템을 구축하기 위한 프로젝트로,  
카메라 영상 입력을 기반으로 **객체 탐지 → 추적 → 장면 구조화 → 위험도 산출 → 경로 예측 → 정책 추론**  
단계의 전체 파이프라인을 통합한다.  

본 시스템은 Jetson Edge 디바이스에서의 경량 처리와 서버 측 고성능 연산을 위한  
**Edge–Server 분산 구조**를 포함한다.

---

## 📌 1. 시스템 개요

본 프로젝트는 다음 기능을 중심으로 구성된다.

### 1) 실시간 객체 탐지 및 추적
- YOLO-World 기반 오픈 도메인 탐지  
- ByteTrack 기반 다중 객체 추적  
- Edge 환경에서 실시간 처리 가능하도록 최적화

### 2) Scene Graph 기반 장면 구조화
- Lightweight-GNN 또는 Transformer 기반 SGG(Scene Graph Generation) 모델  
- 객체 간 공간·관계 정보를 정형화하여 추론 단계의 입력으로 활용

### 3) SGCL (Static Geometric & Context Logic)
- Scene Graph 기반 정적 기하 분석  
- 거리·방향·패턴 기반 위험도 판단  
- 동작 이상 여부 탐지

### 4) GRIN 기반 동적 경로 예측
- Scene Graph 시퀀스를 입력으로 GRIN 모델 적용  
- 객체의 미래 이동 경로 및 잠재적 충돌 위치 추정  
- SGCL 결과와 결합하여 동적 Risk Score 계산

### 5) LLM Reasoning
- Scene Graph, 위험도, 경로 예측 정보를 종합하여  
  **상황 요약·행동 정책·주의 대상(focus targets)**을 생성  
- 텍스트 기반 및 ROI Feature 기반 VLM 방식 병행 실험

### 6) Jetson Edge–Server 파이프라인
- Edge: 탐지·추적 수행  
- Server: 구조화·위험 분석·추론 수행  
- WebRTC를 통한 영상 및 Metadata 실시간 전송

---

## 🧩 2. 파이프라인 구성

### ▶ 2.1 Object Detection  
**파일:** `infra/detector/yolo_world_adapter.py`  
- YOLO-World v2 기반  
- 출력: [x1, y1, x2, y2, score, class_name]


---

### ▶ 2.2 Object Tracking  
**파일:** `infra/tracker/bytetrack_adapter.py`  
- ByteTrack 기반 다중 객체 추적  
- 출력: `track_id + bbox`

---

### ▶ 2.3 Scene Graph Generation  
**파일:** `lightweight_gnn.py` 또는 Transformer 기반 SGG  
- Node 구성: `id, class, bbox, 위치`  
- Edge 구성: `관계(label), 거리, 방향(spatial order)`  
- GRIN 및 SGCL 입력으로 사용되는 Scene Graph 시퀀스 생성

---

### ▶ 2.4 SGCL  
**Static Geometric & Context Logic**  
- 정적 기하 규칙 기반 분석  
- 객체 간 충돌 위험도, 문맥적 이상행동 판단  
- LLM Reasoning의 입력으로 제공

---

### ▶ 2.5 GRIN Trajectory Prediction  
- Scene Graph 시퀀스 기반 경로 예측  
- 미래 위치 및 잠재적 위험 위치 추산  
- SGCL과 결합하여 종합 위험도 생성

---

### ▶ 2.6 LLM Reasoning  
- 입력: Scene Graph, Risk Score, GRIN trajectory  
- 출력:
- Caption(상황 설명)
- Focus Targets
- Action Plan(정책 제안)
- YOLO-World Prompt(선택적)

---

## 🧠 3. 메모리 구조 (STM / LTM)

AURA는 객체 지속성 유지 및 장면 일관성 향상을 위해  
단기 메모리(STM)와 장기 메모리(LTM)를 구분하여 사용한다.

### 🔹 Short-Term Memory (STM)
- 최근 프레임 기반 일시적 상태 관리  
- Track History, velocity, recent SGG window 포함  
- GRIN 입력 및 추적 안정화에 사용  
- 유지 기간: **수 프레임(약 0.5~1초)**

### 🔹 Long-Term Memory (LTM)
- 장기간 유지되는 객체 정체성 및 패턴 저장  
- Appearance embedding, 관계 이력, 장기 위험 패턴 포함  
- Re-ID 및 의미적 Scene Graph 안정성 제공  
- 유지 기간: **수십 초 ~ 수 분**

---

## 🌐 4. Jetson Edge–Server 구조

### **Edge (Jetson)**
- YOLO-World 기반 실시간 탐지  
- ByteTrack 추적  
- WebRTC를 통해 서버로 영상 및 metadata 전송  

### **Server**
- Scene Graph 생성  
- SGCL 및 GRIN  
- LLM Reasoning  
- React 기반 Web UI 시각화

---

## 🛠️ 5. 실행 방법

### 5.1 환경 구성

#### uv
```bash
uv sync

