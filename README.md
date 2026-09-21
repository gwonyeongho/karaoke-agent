# Karaoke AI Agent

음성 또는 텍스트 명령을 해석해 노래방 기기의 상태를 제어하는 AI 에이전트 프로젝트입니다. 사용자의 명령과 현재 기기 상태를 Qwen에 전달하고 구조화된 결과를 검증한 뒤 웹 제어 화면에 반영합니다.

## 주요 기능

- faster-whisper 기반 음성 명령 인식
- Qwen3 1.7B 기반 자연어 명령 해석
- LangChain 구조화 출력과 Pydantic 데이터 검증
- 음정·템포·볼륨·에코·공간 음향·자막·예약곡 상태 제어
- 모호하거나 지원하지 않는 명령에 대한 기존 상태 유지
- FastAPI 백엔드와 HTML·CSS·JavaScript 웹 제어 화면 연동
- 기기 오른쪽 RAG 사용 도우미에서 프로젝트 문서 검색·근거 기반 답변
- 검색에 사용된 문서 목록과 전체 원문 열람

## RAG 사용 도우미

기기 제어 Agent와 RAG는 서로 분리되어 있습니다. 기존 명령창은 기기 상태를 변경하고, 오른쪽 RAG 패널은 `backend/knowledge`의 프로젝트 사용 안내만 검색합니다. RAG 질문은 예약·음량 등 기기 상태를 변경하지 않습니다.

```text
질문 → bge-m3 임베딩 → Chroma 관련 문서 검색
     → 검색 문맥 + Qwen3 1.7B → 답변·출처·발췌문
```

현재 검색 자료는 재생·예약, 음향·화면 제어, 음성 Agent, 문제 해결을 다룬 자체 작성 문서 4개입니다. 제조사 공식 설명서가 아닙니다. 패널의 **전체 검색 자료 보기**에서는 질문에 사용됐는지와 관계없이 RAG 검색 대상 문서를 모두 확인할 수 있습니다. 답변 출처의 **원문 보기**는 해당 질문에서 검색된 문서를 바로 엽니다. 문서 조회 API는 Ollama가 실행되지 않아도 동작합니다.

## 성능 및 명령 처리 개선

초기 모델은 응답 시간이 길어 실제 조작 과정에서 지연이 발생했습니다. 실시간 사용성을 높이기 위해 최종 모델을 **Qwen3 1.7B**로 변경해 응답 속도를 확보했습니다.

모델 경량화 이후에는 사용자의 의도와 다른 명령이 반환되거나 지원 범위를 벗어난 값이 생성될 가능성을 보완했습니다.

- LangChain의 `with_structured_output`을 적용해 모델 응답을 `KaraokeMachine` 스키마로 제한
- Pydantic으로 필드 타입과 음정·템포·볼륨 등의 허용 범위 검증
- 현재 기기 상태를 명령과 함께 전달해 변경되지 않은 상태 유지
- 모호하거나 지원하지 않는 요청은 기존 상태를 유지하도록 규칙 설정
- `temperature=0`을 적용해 같은 조건에서 출력 변동 축소

LangChain은 모델 자체의 추론 성능을 높이는 용도가 아니라 구조화된 호출과 출력 연결에 사용했습니다. 경량 모델의 빠른 응답을 유지하면서 Pydantic과 명령 규칙으로 실행 안정성을 보완했습니다.

## 현재 처리 흐름

```text
음성 녹음
→ faster-whisper 음성 인식
→ 현재 상태와 명령을 Qwen에 전달
→ LangChain 구조화 출력
→ Pydantic 타입·허용값·범위 검증
→ 검증된 상태를 웹 화면에 반영
```

전체 상태를 LLM이 다시 생성하는 현재 구조의 한계를 보완하기 위해 LangGraph 기반 전환을 계획하고 있습니다. 자세한 내용은 [LangGraph 전환 계획](./docs/LANGGRAPH_PLAN.md)에서 확인할 수 있습니다.

## 기술 스택

- **Backend:** Python, FastAPI, Pydantic
- **AI:** LangChain, ChatOllama, Ollama, Qwen3 1.7B, bge-m3
- **RAG:** Chroma, Markdown knowledge corpus
- **Speech:** faster-whisper
- **Frontend:** HTML, CSS, Vanilla JavaScript

## 프로젝트 구조

```text
.
├── backend/
│   ├── knowledge/             # RAG가 검색하고 UI에서 공개하는 원문
│   ├── rag/                   # 로더·검색·답변·문서 조회 API
│   ├── __init__.py
│   └── main.py                # FastAPI, STT, LLM 명령 처리와 출력 검증
├── frontend/
│   ├── index.html             # 노래방 제어 화면
│   ├── app.js                 # 상태 관리와 API·음성 입력 연동
│   └── style.css              # 화면 스타일
├── docs/
│   └── LANGGRAPH_PLAN.md      # LangGraph 전환 구현 계획
├── README.md
├── TODO.md
└── requirements.txt
```

## 실행 방법

### 1. Ollama 및 모델 준비

Ollama를 설치한 뒤 생성 모델과 한국어 검색용 임베딩 모델을 내려받습니다.

```bash
ollama pull qwen3:1.7b
ollama pull bge-m3
```

### 2. Python 의존성 설치

```bash
python -m venv .venv
```

Windows Git Bash:

```bash
source .venv/Scripts/activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

### 3. API 서버 실행

프로젝트 루트에서 실행합니다.

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API 문서는 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.

RAG 관련 환경변수는 `RAG_CHAT_MODEL`, `RAG_EMBED_MODEL`, `RAG_TOP_K`, `RAG_SCORE_THRESHOLD`입니다. 최초 질문 시 `.rag_index`에 로컬 Chroma 인덱스가 생성되며 지식 문서 내용이 바뀌면 새 버전의 컬렉션을 사용합니다.

### 4. 웹 화면 실행

새 터미널에서 프로젝트 루트를 기준으로 실행합니다.

```bash
python -m http.server 5500 --directory frontend
```

브라우저에서 `http://127.0.0.1:5500`을 엽니다.

## STT 환경 설정

기본 설정은 CPU 환경에서 faster-whisper의 `small` 모델과 `int8` 연산을 사용합니다.

```bash
export WHISPER_MODEL=small
export WHISPER_DEVICE=cpu
export WHISPER_COMPUTE=int8
```

GPU 환경에서는 장치와 연산 방식을 실행 환경에 맞게 변경할 수 있습니다.

## 문서

- [개선 TODO](./TODO.md)
- [LangGraph 전환 계획](./docs/LANGGRAPH_PLAN.md)
- [LangChain 및 RAG 기능·처리 흐름](./docs/RAG_LANGCHAIN_FLOW.md)
- [RAG 사이드 도우미 구현 계획](./.hermes/plans/2026-09-21_135002-rag-side-assistant.md)
- [RAG 검색 자료 열람 설계](./docs/RAG_DOCUMENT_VIEWER.md)
