AURA: Autonomous Understanding Reactive & Attention

YOLO-World · ByteTrack · Scene Graph Generation · SGCL · GRIN · LLM Reasoning

AURA는 실시간 로봇 인지 시스템 개발을 목적으로 하는 프로젝트이며,
카메라 입력으로부터 객체 탐지, 추적, 장면 구조화, 위험도 분석, 경로 예측, 정책 추론까지
일련의 단계를 통합한 인지–추론 파이프라인을 구현한다.

프로젝트는 Jetson 기반 Edge 디바이스에서의 경량 탐지·추적과
서버에서의 고차원 구조화·추론 작업을 분산 처리하도록 설계되어 있다.

1. 시스템 개요

본 프로젝트는 다음의 주요 기능을 포함한다.

실시간 객체 탐지 및 추적
YOLO-World 기반 탐지 모델과 ByteTrack 기반 다중 객체 추적기를 사용하여
비정형 환경에서도 객체의 변화와 지속성을 안정적으로 확보한다.

Scene Graph 기반 장면 구조화
Lightweight-GNN 또는 Transformer 기반 Scene Graph Generation(SGG) 모델을 이용하여
객체 간 관계(공간적 위치, 상호작용 등)를 구조화한다.
이는 후속 단계의 위험 분석과 정책 추론의 입력으로 사용된다.

SGCL (Static Geometric & Context Logic)
Scene Graph를 입력으로 정적 기하 분석을 수행하고
객체 간의 거리, 방향, 속성 기반 논리 규칙을 적용하여 정적 위험도를 산출한다.

GRIN 기반 동적 경로 예측
시간 축 Scene Graph 시퀀스를 기반으로 GRIN(Gaussian Recurrent Interaction Network)을 적용하여
객체의 미래 이동 경로를 예측한다.
정적 위험도와 결합하여 동적 위험도를 갱신한다.

LLM Reasoning
Scene Graph, 위험도 평가 결과, GRIN trajectory 정보를 종합하여
행동 전략, 상황 설명, 주의 대상(focus targets)을 생성한다.
텍스트 기반 구조뿐 아니라 ROI feature 기반 VLM 입력 방식도 실험 중이다.

Jetson Edge–Server 분산 스트리밍 구조
Edge 디바이스에서는 탐지·추적을 수행하고,
서버에서는 Scene Graph 생성, SGCL, GRIN, LLM 등을 처리한다.
WebRTC를 사용하여 영상과 metadata를 실시간 전송한다.

2. 파이프라인 구성
2.1 Object Detection

파일: infra/detector/yolo_world_adapter.py

YOLO-World v2 기반 오픈 도메인 객체 탐지

출력 형식:

[x1, y1, x2, y2, score, class_name]

2.2 Object Tracking

파일: infra/tracker/bytetrack_adapter.py

ByteTrack 기반 다중 객체 추적

단일 프레임의 detection을 시간적 연속성에 기반해 통합

출력: track_id + bounding box

2.3 Scene Graph Generation

파일: lightweight_gnn.py 또는 SGTR/MotifNet 기반 SGG 모듈

출력 요소:

Node: 객체 ID, class, bbox, 위치 정보

Edge: 관계(label), 거리, 방향, 공간 순서(spatial ordering)

Scene Graph 시퀀스는 GRIN 및 SGCL의 입력으로 사용된다.

2.4 SGCL (Static Geometric & Context Logic)

Scene Graph의 기하적 및 문맥적 규칙을 평가

객체 간 충돌 가능성, 위험도, 비정상적 패턴 등을 정적으로 분석

결과는 GRIN과 LLM Reasoning 단계에서 사용됨

2.5 GRIN Trajectory Prediction

입력: Scene Graph 시퀀스 및 단기 움직임 정보(STM)

출력: 미래 좌표, 잠재적 충돌 위치, 동적 위험도

SGCL 결과와 결합하여 종합 Risk Score 산출

2.6 LLM Reasoning

입력 정보:

Scene Graph (정적 + 동적)

위험도

미래 이동 경로

출력 정보:

Caption (상황 요약)

Focus Targets

행동 전략(Action Plan)

YOLO-World prompts(선택적)

3. 메모리 구조
3.1 Short-Term Memory (STM)

STM은 최근 수 프레임 동안 유지되는 단기 상태 정보를 저장한다.

주요 구성 요소:

Track History (위치·속도)

최근 Scene Graph Window

일시적 객체 상태 (visibility, temporal stability)

GRIN에 필요한 단기 움직임 벡터

역할:

추적 안정성 유지

Scene Graph의 프레임 간 일관성 확보

GRIN trajectory prediction의 입력 제공

3.2 Long-Term Memory (LTM)

LTM은 장기적으로 객체의 정체성과 의미적 패턴을 저장한다.

주요 구성 요소:

Appearance embedding (YOLO-World ROI feature 또는 TinyReIDHead)

Entity ID persistence

관계 이력 (scene graph interaction history)

장기 위험 행동 패턴

temporal stability score

역할:

재등장 객체 식별(Re-ID)

Scene Graph의 의미적 안정성 확보

LLM Reasoning에서 entity-level context 제공

4. Jetson Edge–Server 구조
Edge (Jetson)

YOLO-World inference

ByteTrack tracking

영상 및 metadata 압축 후 송신

저지연 WebRTC 기반 전송

Server

Scene Graph 생성

SGCL

GRIN trajectory prediction

LLM reasoning

React 웹 인터페이스로 결과 시각화
