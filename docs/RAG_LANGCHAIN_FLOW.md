# LangChain 및 RAG 기능 구조

## 1. 전체 구조

노래방 AI Agent는 두 개의 처리 흐름으로 구성된다.

```text
사용자 입력
├─ 기기 제어 명령 → LangChain 기반 기기 제어
└─ 사용법 질문   → RAG 기반 문서 검색·답변
```

두 흐름은 서로 분리되어 있다.

- 기기 제어 명령은 노래방 상태를 변경한다.
- RAG 질문은 문서를 검색해 답변하며 기기 상태를 변경하지 않는다.

---

## 2. LangChain 기반 기기 제어

### 기능

- 텍스트 명령 처리
- 음성 명령의 STT 결과 처리
- 현재 기기 상태와 사용자 명령을 Qwen3 1.7B에 전달
- LLM 출력을 `KaraokeMachine` 구조로 변환
- Pydantic을 통한 타입·허용값·범위 검증
- 검증된 상태를 웹 제어 화면에 반영

### 처리 흐름

```text
사용자 텍스트 명령
→ FastAPI /api/v1/commands
→ 현재 기기 상태와 명령 구성
→ LangChain ChatOllama
→ Qwen3 1.7B 명령 해석
→ LangChain 구조화 출력
→ Pydantic 상태 검증
→ 변경된 기기 상태 반환
→ 웹 화면 반영
```

음성 명령은 STT 과정이 먼저 추가된다.

```text
사용자 음성
→ MediaRecorder 녹음
→ FastAPI /api/v1/stt
→ faster-whisper 음성 인식
→ 인식된 텍스트
→ LangChain 기반 기기 제어 흐름
```

### LangChain의 역할

- `ChatOllama`를 통해 Qwen3 1.7B 호출
- 시스템 프롬프트와 현재 상태, 사용자 명령 전달
- `with_structured_output`으로 LLM 출력을 Pydantic 모델에 연결

LangChain은 기기 상태를 직접 검증하지 않는다. 타입·허용값·범위 검증은 Pydantic이 담당한다.

---

## 3. RAG 기반 사용 도우미

### 기능

- 노래방 사용법 질문 처리
- 오류 해결 방법 검색
- 관련 문서 조각 검색
- 검색된 문서만 근거로 답변 생성
- 답변에 문서명과 발췌문 표시
- 검색에 사용되는 전체 문서 목록 및 원문 열람
- 관련 근거가 부족하면 답변을 생성하지 않고 자료 부족 안내

### 처리 흐름

```text
사용자 질문
→ FastAPI /api/v1/rag/query
→ 질문 Pydantic 검증
→ OllamaEmbeddings의 bge-m3로 질문 임베딩
→ Chroma에서 관련 문서 조각 검색
→ 검색 거리 점수 기준으로 관련성 확인
   ├─ 관련 문서 있음
   │  → 검색 문맥과 질문을 Qwen3 1.7B에 전달
   │  → 문서 기반 답변 생성
   │  → 답변·문서명·발췌문 반환
   └─ 관련 문서 없음
      → "제공된 문서에서 확인할 수 없습니다." 반환
```

### 문서 준비 흐름

```text
backend/knowledge의 Markdown 문서
→ 문서 제목·섹션 단위 분리
→ 일정 길이의 검색 조각으로 분할
→ bge-m3 임베딩
→ Chroma 컬렉션 저장
```

문서 내용의 해시로 지식 버전을 생성한다. 문서가 변경되면 다음 RAG 질문에서 변경된 버전에 맞는 Chroma 컬렉션을 생성하거나 불러온다.

### RAG API

```text
POST /api/v1/rag/query
- 질문을 검색하고 근거 기반 답변 반환

GET /api/v1/rag/documents
- RAG 검색 대상 전체 문서 목록 반환

GET /api/v1/rag/documents/{document_id}
- 선택한 검색 문서의 전체 원문 반환

GET /api/v1/rag/health
- 문서 수, 지식 버전, 활성 인덱스 버전, 모델 설정 반환
```

### 검색 자료

```text
backend/knowledge/
├─ playback_and_reservation.md
├─ sound_and_display_controls.md
├─ voice_and_agent_guide.md
└─ troubleshooting.md
```

오른쪽 RAG 패널의 `전체 검색 자료 보기`에서 검색 대상 문서 전체를 확인할 수 있다. 답변 아래 `참조 문서`에는 해당 질문에서 실제 검색된 문서와 발췌문만 표시된다.

---

## 4. LangChain과 RAG의 연결

RAG는 문서를 검색해 답변 근거를 제공하는 처리 구조이고, LangChain은 이 구조에서 모델과 임베딩을 연결하는 데 사용된다.

```text
RAG 처리 구조
├─ LangChain OllamaEmbeddings → 문서·질문 임베딩
├─ Chroma                     → 관련 문서 검색
└─ LangChain ChatOllama       → 검색 문맥 기반 답변 생성
```

기존 기기 제어와 RAG 모두 LangChain의 연동 기능을 사용하지만 처리 목적과 API는 분리되어 있다.

```text
기기 제어
/api/v1/commands
→ 상태 변경 가능

RAG
/api/v1/rag/query
→ 문서 검색과 답변만 수행
→ 상태 변경 없음
```

---

## 5. 관련 파일

```text
backend/main.py
- 기존 기기 제어, STT, RAG 라우터 연결

backend/rag/api.py
- RAG 질문·문서 목록·원문·상태 API

backend/rag/runtime.py
- bge-m3, Chroma, Qwen3 1.7B 연결과 인덱스 관리

backend/rag/service.py
- 문서 검색, 관련성 확인, 답변 생성

backend/rag/loader.py
- Markdown 문서 로딩과 분할

backend/rag/documents.py
- 전체 문서 목록, 원문, 문서 버전 관리

backend/rag/schemas.py
- RAG 요청·답변·출처 데이터 검증

backend/knowledge/
- RAG가 검색하는 원문 문서

frontend/index.html
frontend/app.js
frontend/style.css
- 오른쪽 RAG 패널, 답변·출처 표시, 전체 문서 열람 화면
```
