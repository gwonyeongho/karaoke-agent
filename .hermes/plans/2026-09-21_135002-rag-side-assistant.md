# Karaoke RAG Side Assistant Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 기존 노래방 기기 오른쪽에 사용법·문제 해결 질문을 받는 RAG 도우미를 추가하고 검색된 문서 근거와 함께 답변한다.

**Architecture:** 기존 자연어 기기 제어 흐름과 RAG 질의 흐름을 분리한다. 기기 제어는 현재 `/api/v1/commands`를 유지하고 RAG 패널은 `/api/v1/rag/query`만 호출한다. 백엔드는 프로젝트 내 노래방 사용 안내 문서를 분할·임베딩하여 Chroma에 저장하고 관련 문서를 검색한 뒤 Qwen3 1.7B가 검색 내용만 근거로 답변하도록 한다.

**Tech Stack:** FastAPI, Pydantic, LangChain, `ChatOllama`, `OllamaEmbeddings`, Chroma, `langchain-text-splitters`, HTML/CSS/Vanilla JavaScript, pytest

---

## 1. 확정 범위

### 사용자 화면

- 기존 기기 오른쪽 숫자패드는 제거하지 않는다.
- 전체 화면을 `노래방 기기 + RAG 사용 도우미` 2열로 구성한다.
- RAG 패널은 다음 요소를 가진다.
  - 제목: `RAG 사용 도우미`
  - 설명: `사용법과 오류 해결 방법을 문서에서 찾아 답합니다.`
  - 질문 입력창
  - `검색` 버튼
  - 로딩 상태
  - 답변 영역
  - 참조 문서명과 검색된 문장 영역
  - 자료 부족 안내
- 모바일과 좁은 화면에서는 RAG 패널을 기기 아래로 내린다.

### 지식 범위

초기 버전은 프로젝트가 직접 작성한 다음 문서만 검색한다.

- 노래방 기본 사용법
- 곡 재생·예약·취소 방법
- 음정·템포·마이크·에코 조절 방법
- 자막과 MR 기능 설명
- 마이크 인식 실패·API 연결 실패 등 데모 문제 해결

실제 TJ B80 공식 설명서를 확보하지 않은 상태이므로 공식 매뉴얼이라고 표현하지 않는다. 이후 사용자가 제공한 문서로 교체할 수 있도록 출처 경로를 분리한다.

### 안전 경계

- RAG 답변은 기기 상태를 변경하지 않는다.
- RAG 질문을 기존 `/api/v1/commands`에 전달하지 않는다.
- 검색 근거가 부족하면 추측하지 않고 `제공된 문서에서 확인할 수 없습니다.`라고 답한다.
- 답변에 참조 문서명과 검색 근거를 함께 반환한다.
- 프롬프트에 없는 사용법을 모델의 일반 지식으로 보완하지 않는다.

---

## 2. 목표 처리 흐름

```text
사용자 질문
→ POST /api/v1/rag/query
→ 질문 Pydantic 검증
→ OllamaEmbeddings로 질문 임베딩
→ Chroma에서 관련 문서 조각 검색
→ 검색 결과의 관련성 확인
   ├─ 근거 있음 → 검색 문맥 + 질문을 Qwen3 1.7B에 전달
   └─ 근거 부족 → 답변 생성 없이 자료 부족 응답
→ 답변 + 문서명 + 검색 문장 반환
→ 오른쪽 RAG 패널에 표시
```

기기 제어는 기존 흐름을 그대로 유지한다.

```text
사용자 명령
→ POST /api/v1/commands
→ Qwen 명령 해석
→ 구조화 출력
→ Pydantic 검증
→ 기기 상태 반영
```

---

## 3. 예정 파일 구조

```text
1516024-master/
├─ backend/
│  ├─ __init__.py
│  ├─ main.py                         # RAG API 라우트 연결
│  ├─ rag/
│  │  ├─ __init__.py
│  │  ├─ config.py                    # 모델·문서·검색 설정
│  │  ├─ schemas.py                   # 요청·응답·출처 모델
│  │  ├─ loader.py                    # Markdown 로딩·분할
│  │  └─ service.py                   # 임베딩·검색·답변 생성
│  └─ knowledge/
│     ├─ karaoke_guide.md             # 기능별 사용법
│     └─ troubleshooting.md           # 오류·문제 해결
├─ frontend/
│  ├─ index.html                      # 오른쪽 RAG 패널
│  ├─ style.css                       # 2열·반응형 레이아웃
│  └─ app.js                          # RAG API 호출·렌더링
├─ tests/
│  ├─ test_rag_loader.py
│  ├─ test_rag_service.py
│  └─ test_rag_api.py
├─ .gitignore                         # 로컬 벡터 인덱스 제외
├─ requirements.txt
└─ README.md
```

---

### Task 1: RAG 의존성과 환경설정 추가

**Objective:** Chroma와 Ollama 임베딩을 재현 가능한 설정으로 추가한다.

**Files:**
- Modify: `requirements.txt`
- Create: `backend/rag/__init__.py`
- Create: `backend/rag/config.py`
- Modify: `.gitignore`

**Step 1: 의존성 정의**

`requirements.txt`에 다음 패키지를 명시한다.

```text
langchain-chroma
langchain-text-splitters
chromadb
pytest
httpx
```

`OllamaEmbeddings`는 현재 설치된 `langchain-ollama`를 사용한다.

**Step 2: 설정값 작성**

`backend/rag/config.py`에 다음 설정을 둔다.

```python
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
PERSIST_DIR = BASE_DIR.parent / ".rag_index"

RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "qwen3:1.7b")
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.65"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
```

**Step 3: 생성 파일 제외**

`.gitignore`에 다음을 추가한다.

```text
.rag_index/
__pycache__/
.pytest_cache/
```

**Step 4: 설치 검증**

Run:

```bash
uv pip install -r requirements.txt
python -c "from langchain_chroma import Chroma; from langchain_ollama import OllamaEmbeddings"
```

Expected: import 오류 없이 종료 코드 0.

**Step 5: 모델 준비**

Run:

```bash
ollama pull nomic-embed-text
ollama list
```

Expected: `bge-m3`와 기존 `qwen3:1.7b`가 목록에 표시됨.

**Step 6: Commit**

```bash
git add requirements.txt .gitignore backend/rag
 git commit -m "chore: add RAG dependencies and configuration"
```

---

### Task 2: 검색 대상 지식 문서 작성

**Objective:** RAG가 답변할 수 있는 범위와 근거를 명시한 문서를 만든다.

**Files:**
- Create: `backend/knowledge/karaoke_guide.md`
- Create: `backend/knowledge/troubleshooting.md`

**Step 1: 사용법 문서 작성**

`karaoke_guide.md`를 기능 단위 제목으로 구성한다.

```markdown
# 노래방 제어 패널 사용 안내

## 곡 재생
곡 번호를 입력하고 시작 버튼을 누르면 해당 번호를 재생합니다.
곡 번호를 입력하지 않고 시작 버튼을 누르면 예약 목록의 첫 곡을 재생합니다.

## 예약
곡 번호를 입력한 뒤 예약 버튼을 누르면 예약 목록의 마지막에 추가합니다.
우선예약 버튼은 곡을 예약 목록의 앞에 추가합니다.

## 음정과 템포
음정은 -6부터 6까지 조절합니다.
템포는 -5부터 5까지 조절합니다.

## 음량과 음향
마이크·반주·멜로디·에코는 0부터 100까지 조절합니다.

## MR과 자막
MR ON은 보컬 제거를 켭니다.
지원 자막은 한국어·영어·일본어·중국어입니다.
```

실제 코드에 존재하는 기능과 범위만 기록한다. 구현되지 않은 기능은 문서에 추가하지 않는다.

**Step 2: 문제 해결 문서 작성**

다음 항목을 코드와 README 기준으로 작성한다.

- API 연결 실패
- 음성 인식 결과가 비어 있음
- 마이크 권한 거부
- 허용 범위를 벗어난 명령
- 지원하지 않는 명령
- Ollama 모델 미설치
- ffmpeg 또는 Whisper 실행 문제

**Step 3: 문서 사실 검사**

- `backend/main.py`의 필드 범위와 문서 값 비교
- 실제 프론트 버튼과 문서 기능 비교
- 공식 TJ 매뉴얼이라는 표현이 없는지 확인

**Step 4: Commit**

```bash
git add backend/knowledge
 git commit -m "docs: add karaoke RAG knowledge sources"
```

---

### Task 3: 문서 로딩과 청킹 구현

**Objective:** Markdown 문서를 안정적으로 읽고 출처 메타데이터를 보존한 문서 조각으로 변환한다.

**Files:**
- Create: `backend/rag/loader.py`
- Test: `tests/test_rag_loader.py`

**Step 1: 실패 테스트 작성**

```python
from pathlib import Path
from backend.rag.loader import load_and_split_documents


def test_load_and_split_documents_keeps_source_name(tmp_path: Path):
    (tmp_path / "guide.md").write_text(
        "# 예약\n곡 번호를 입력하고 예약 버튼을 누릅니다.",
        encoding="utf-8",
    )

    chunks = load_and_split_documents(tmp_path, chunk_size=80, overlap=10)

    assert chunks
    assert chunks[0].metadata["source"] == "guide.md"
    assert "예약" in chunks[0].page_content
```

빈 디렉터리와 `.md` 외 파일 무시 테스트도 추가한다.

**Step 2: 실패 확인**

Run:

```bash
pytest tests/test_rag_loader.py -v
```

Expected: `backend.rag.loader` 미구현으로 FAIL.

**Step 3: 최소 구현**

- `Path.glob("*.md")`로 정렬된 파일 목록 사용
- UTF-8 읽기
- `Document(page_content=..., metadata={"source": path.name})`
- `RecursiveCharacterTextSplitter` 사용
- 구분자는 `\n## `, `\n### `, `\n\n`, `\n`, ` ` 순서
- 빈 문서 제외

**Step 4: 통과 확인**

Run:

```bash
pytest tests/test_rag_loader.py -v
```

Expected: 모든 테스트 PASS.

**Step 5: Commit**

```bash
git add backend/rag/loader.py tests/test_rag_loader.py
 git commit -m "feat: load and split RAG knowledge documents"
```

---

### Task 4: RAG 요청·응답 스키마 정의

**Objective:** 질문과 출처 응답을 Pydantic으로 검증한다.

**Files:**
- Create: `backend/rag/schemas.py`
- Test: `tests/test_rag_schemas.py`

**Step 1: 실패 테스트 작성**

검증 조건:

- 질문 앞뒤 공백 제거
- 빈 질문 거부
- 질문 최대 500자
- `top_k`는 1~5
- 답변에 `grounded`와 `sources` 포함

**Step 2: 모델 구현**

```python
class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=5)

class RagSource(BaseModel):
    source: str
    excerpt: str
    score: float | None = None

class RagQueryResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[RagSource] = Field(default_factory=list)
```

질문 공백 제거 validator를 추가한다.

**Step 3: 검증**

Run:

```bash
pytest tests/test_rag_schemas.py -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add backend/rag/schemas.py tests/test_rag_schemas.py
 git commit -m "feat: define validated RAG API schemas"
```

---

### Task 5: 검색 및 근거 기반 답변 서비스 구현

**Objective:** 문서를 검색하고 검색 근거만 사용해 답변하는 핵심 서비스를 구현한다.

**Files:**
- Create: `backend/rag/service.py`
- Test: `tests/test_rag_service.py`

**Step 1: 서비스 테스트를 의존성 없이 작성**

가짜 검색기와 가짜 LLM을 주입할 수 있게 설계한다.

테스트 항목:

1. 관련 문서가 있으면 문맥과 질문을 LLM에 전달한다.
2. 반환 출처에 파일명과 발췌문이 포함된다.
3. 검색 결과가 없으면 LLM을 호출하지 않는다.
4. 점수가 임계값을 넘으면 자료 부족으로 응답한다.
5. 여러 문서가 검색돼도 `top_k`를 넘지 않는다.
6. 모델 오류가 내부 예외로 변환된다.

Chroma의 거리 점수는 낮을수록 유사하다는 점을 테스트와 코드 주석에 명시한다. 환경에 따라 점수 의미가 달라질 수 있으므로 실제 임베딩 결과를 확인한 뒤 기본 임계값을 조정한다.

**Step 2: 실패 확인**

Run:

```bash
pytest tests/test_rag_service.py -v
```

Expected: 서비스 미구현으로 FAIL.

**Step 3: 벡터 저장소 생성**

- `OllamaEmbeddings(model=RAG_EMBED_MODEL)` 사용
- 문서의 내용 해시를 계산해 인덱스 버전을 판단
- 지식 문서가 변경되면 인덱스 재생성
- 지식 문서가 없으면 명확한 초기화 오류 반환
- 서비스 객체는 첫 RAG 요청 시 lazy initialization

**Step 4: 답변 프롬프트 작성**

```text
너는 노래방 제어 패널 사용 도우미다.
반드시 제공된 검색 문맥만 사용한다.
문맥에 없는 내용을 추측하거나 일반 지식으로 보완하지 않는다.
답을 찾을 수 없으면 "제공된 문서에서 확인할 수 없습니다."라고 답한다.
기기 상태를 변경했다고 말하지 않는다.
짧고 명확한 한국어로 답한다.

[검색 문맥]
{context}

[질문]
{question}
```

**Step 5: 자료 부족 응답 고정**

검색 결과가 없거나 관련성 기준을 만족하지 못하면 모델을 호출하지 않고 다음을 반환한다.

```python
RagQueryResponse(
    answer="제공된 문서에서 확인할 수 없습니다.",
    grounded=False,
    sources=[],
)
```

**Step 6: 통과 확인**

Run:

```bash
pytest tests/test_rag_service.py -v
```

Expected: PASS.

**Step 7: 실제 검색 스모크 테스트**

Ollama 실행 후 다음 질문으로 확인한다.

- `음정은 어디까지 올릴 수 있어?`
- `예약한 곡을 취소하려면 어떻게 해?`
- `마이크 인식이 안 되면 뭘 확인해야 해?`
- `비행기 표를 예약해줘` → 자료 부족

**Step 8: Commit**

```bash
git add backend/rag/service.py tests/test_rag_service.py
 git commit -m "feat: implement grounded karaoke RAG service"
```

---

### Task 6: FastAPI RAG 엔드포인트 연결

**Objective:** RAG 질의를 별도 API로 제공하고 기존 기기 제어 API와 격리한다.

**Files:**
- Modify: `backend/main.py`
- Test: `tests/test_rag_api.py`

**Step 1: 실패 테스트 작성**

테스트에서는 실제 Ollama·Whisper를 실행하지 않고 RAG 서비스 의존성을 대체한다.

검증 항목:

- `POST /api/v1/rag/query` 정상 응답
- 빈 질문 422
- 500자를 넘는 질문 422
- RAG 초기화 실패 시 503과 사용자용 메시지
- RAG 질의 전후 기기 상태가 변경되지 않음
- 기존 `/api/v1/commands` 경로가 유지됨

현재 `backend/main.py`가 import 시 Whisper 모델을 생성하므로 테스트가 느리거나 실패할 수 있다. RAG API 테스트 전에 Whisper 모델도 최초 STT 요청 시 생성하는 lazy getter로 옮긴다. 동작은 변경하지 않고 초기화 시점만 바꾼다.

**Step 2: API 구현**

```python
@app.post("/api/v1/rag/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest):
    try:
        return get_rag_service().answer(request.question, request.top_k)
    except RagUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
```

선택적으로 상태 확인용 엔드포인트를 추가한다.

```text
GET /api/v1/rag/health
```

반환 정보는 비밀값 없이 문서 수·인덱스 준비 여부·모델명만 포함한다.

**Step 3: 통과 확인**

Run:

```bash
pytest tests/test_rag_api.py -v
pytest -v
```

Expected: 전체 PASS.

**Step 4: 수동 API 검증**

Run:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"예약은 어떻게 해?","top_k":4}'
```

Expected:

- `grounded: true`
- 사용법 답변
- `karaoke_guide.md` 출처
- 발췌문 포함

**Step 5: Commit**

```bash
git add backend/main.py tests/test_rag_api.py
 git commit -m "feat: expose isolated RAG query API"
```

---

### Task 7: 기기 오른쪽 RAG 패널 추가

**Objective:** 데스크톱에서 노래방 기기 오른쪽에 RAG 인터페이스를 배치한다.

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/style.css`

**Step 1: 전체 레이아웃 래퍼 추가**

기존 `.machine-wrapper`와 새 `<aside class="rag-panel">`을 다음 구조로 감싼다.

```html
<div class="app-workspace">
  <div class="machine-wrapper">
    <!-- 기존 기기 전체 -->
  </div>

  <aside class="rag-panel" aria-labelledby="ragTitle">
    <div class="rag-panel-head">
      <span class="rag-label">RAG</span>
      <h2 id="ragTitle">사용 도우미</h2>
      <p>사용법과 오류 해결 방법을 문서에서 찾아 답합니다.</p>
    </div>

    <form id="ragForm" class="rag-form">
      <label for="ragQuestion">질문</label>
      <textarea id="ragQuestion" maxlength="500"
        placeholder="예: 예약한 곡을 취소하려면 어떻게 해?"></textarea>
      <button id="btnRagAsk" class="btn" type="submit">문서 검색</button>
    </form>

    <div id="ragStatus" class="rag-status" role="status"></div>
    <section id="ragAnswer" class="rag-answer" aria-live="polite"></section>
    <section class="rag-sources">
      <h3>참조 문서</h3>
      <div id="ragSourceList"></div>
    </section>
  </aside>
</div>
```

**Step 2: 데스크톱 레이아웃 구현**

```css
.app-workspace {
  width: min(1520px, 100%);
  display: grid;
  grid-template-columns: minmax(0, 1100px) minmax(300px, 360px);
  gap: 24px;
  align-items: start;
}

.rag-panel {
  position: sticky;
  top: 20px;
  min-height: 620px;
}
```

기존 기기 디자인과 어울리는 어두운 패널·청록색 강조색을 사용한다. 답변과 출처는 시각적으로 분리한다.

**Step 3: 반응형 처리**

```css
@media (max-width: 1280px) {
  .app-workspace {
    grid-template-columns: 1fr;
  }

  .rag-panel {
    position: static;
    min-height: auto;
  }
}
```

현재 기기 자체가 고정 3열이므로 1100px 이하에서 기기 내부 가로 넘침도 확인하고 필요한 경우 스케일링이 아닌 내부 열 재배치를 별도로 적용한다.

**Step 4: 정적 검증**

- ID 중복 없음
- label과 textarea 연결
- Enter+Ctrl 또는 버튼으로 제출 가능하도록 form 사용
- 답변 영역 `aria-live` 적용

**Step 5: Commit**

```bash
git add frontend/index.html frontend/style.css
 git commit -m "feat: add RAG assistant beside karaoke machine"
```

---

### Task 8: 프론트엔드 RAG API 연동

**Objective:** 오른쪽 패널의 질문을 RAG API로 보내고 답변과 출처를 안전하게 표시한다.

**Files:**
- Modify: `frontend/app.js`
- Test: `tests/frontend/rag-ui.test.js` 또는 브라우저 수동 테스트

**Step 1: DOM 요소와 엔드포인트 정의**

```javascript
const RAG_ENDPOINT = "/api/v1/rag/query";
const ragForm = $("ragForm");
const ragQuestion = $("ragQuestion");
const btnRagAsk = $("btnRagAsk");
const ragStatus = $("ragStatus");
const ragAnswer = $("ragAnswer");
const ragSourceList = $("ragSourceList");
```

**Step 2: XSS 안전 렌더링 함수 작성**

- 답변과 발췌문은 `innerHTML`이 아닌 `textContent` 사용
- 문서별 source card를 DOM API로 생성
- 긴 발췌문은 CSS로 줄바꿈

**Step 3: API 호출 함수 작성**

```javascript
async function askRag(question) {
  const res = await fetch(API_BASE + RAG_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: question.trim(), top_k: 4 }),
  });

  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
```

**Step 4: 상태 처리**

- 빈 질문: API 호출 금지
- 요청 중: 버튼 비활성화와 `문서 검색 중...`
- 성공: 답변과 출처 표시
- `grounded=false`: 자료 부족 상태 강조
- 실패: `RAG 서버와 임베딩 모델 상태를 확인하세요.`
- 완료 후 버튼 활성화

**Step 5: 기존 상태와의 격리 확인**

질의 전후 `originStatus`가 동일한지 확인한다. `askRag`는 `sendCommand`와 `lastResponse`를 호출하거나 수정하지 않는다.

**Step 6: JavaScript 문법 검사**

Run:

```bash
node --check frontend/app.js
```

Expected: 종료 코드 0.

**Step 7: Commit**

```bash
git add frontend/app.js
 git commit -m "feat: connect RAG assistant to retrieval API"
```

---

### Task 9: 통합 테스트와 검색 품질 조정

**Objective:** 실제 Ollama 모델과 브라우저를 사용해 검색·답변·UI를 검증한다.

**Files:**
- Modify if needed: `backend/rag/config.py`
- Modify if needed: `backend/knowledge/*.md`
- Modify if needed: `frontend/style.css`

**Step 1: 백엔드 실행**

```bash
python -m uvicorn backend.main:app --reload
```

Expected: `/docs` 접근 가능하고 RAG 초기화 오류가 없음.

**Step 2: 프론트 실행**

```bash
python -m http.server 5500 -d frontend
```

Expected: `http://127.0.0.1:5500`에서 화면 표시.

**Step 3: 정답 질문 검증**

| 질문 | 기대 결과 |
|---|---|
| 예약은 어떻게 해? | 예약 버튼과 목록 추가 방법 안내 |
| 음정 범위는? | -6~6 안내 |
| 영어 자막을 켤 수 있어? | ENG 자막 지원 안내 |
| 마이크 인식이 안 돼 | 권한·무음·서버/모델 확인 안내 |

각 응답에 문서명과 관련 발췌문이 포함되어야 한다.

**Step 4: 답변 거부 검증**

| 질문 | 기대 결과 |
|---|---|
| 오늘 날씨가 어때? | 문서에서 확인할 수 없음 |
| 비행기 표를 예약해줘 | 문서에서 확인할 수 없음 |
| 이 노래방의 실제 가격은? | 문서에 없으면 확인할 수 없음 |

모델의 일반 지식으로 답하면 실패다.

**Step 5: 기기 상태 격리 검증**

1. 현재 `origin_status JSON` 저장
2. RAG 질문 3개 전송
3. JSON 다시 비교
4. 값이 완전히 동일해야 함

**Step 6: 데스크톱 UI 검증**

- 1440×900에서 기기 오른쪽에 RAG 패널 표시
- 숫자패드와 RAG 패널이 겹치지 않음
- 답변이 길어져도 패널 밖으로 넘치지 않음
- sticky 패널이 하단 콘텐츠를 가리지 않음

**Step 7: 모바일 UI 검증**

- 390×844에서 기기 아래 RAG 패널 표시
- 가로 스크롤 없음
- 입력창·버튼 터치 가능
- 출처 발췌문 줄바꿈 정상

**Step 8: 전체 검사**

```bash
pytest -v
python -m py_compile backend/main.py backend/rag/*.py
node --check frontend/app.js
git diff --check
```

Expected: 모든 테스트와 문법 검사 PASS, whitespace 오류 없음.

**Step 9: Commit**

```bash
git add backend frontend tests
 git commit -m "test: verify grounded RAG assistant flow"
```

---

### Task 10: README와 실행 절차 갱신

**Objective:** 다른 사람이 RAG 기능을 재현하고 현재 구현 범위를 정확히 이해하게 한다.

**Files:**
- Modify: `README.md`
- Modify: `TODO.md`

**Step 1: README 기능 설명**

다음 내용을 추가한다.

- 기기 제어 Agent와 RAG 사용 도우미의 차이
- RAG 처리 흐름 다이어그램
- 검색 문서 경로
- 사용한 임베딩 모델과 벡터 저장소
- 출처 표시 및 자료 부족 처리
- RAG가 기기 상태를 변경하지 않는다는 경계

**Step 2: 실행 방법 갱신**

```bash
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
uv pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
python -m http.server 5500 -d frontend
```

**Step 3: 한계 명시**

- 프로젝트 자체 작성 문서를 사용하는 데모
- 실제 TJ B80 공식 매뉴얼 미포함
- RAG는 정보 검색 전용
- 인덱스는 문서 변경 시 재생성
- 상용 운영 모니터링과 평가 파이프라인은 별도 과제

**Step 4: TODO 정리**

향후 항목:

- 실제 매뉴얼 PDF 로더
- 검색 품질 평가 데이터셋
- 질문 유형 자동 라우팅
- 대화 이력 기반 후속 질문
- 관리자용 문서 갱신 기능

이번 구현에서는 YAGNI 원칙에 따라 추가하지 않는다.

**Step 5: Commit**

```bash
git add README.md TODO.md
 git commit -m "docs: document karaoke RAG assistant"
```

---

## 4. 완료 기준

다음 조건을 모두 충족하면 완료다.

- [ ] 노래방 기기 오른쪽에 RAG 패널이 표시된다.
- [ ] 기존 숫자패드와 기기 제어 기능이 그대로 작동한다.
- [ ] 사용법 질문에 검색 문서 기반 답변이 표시된다.
- [ ] 답변마다 문서명과 발췌문이 표시된다.
- [ ] 문서에 없는 질문은 추측하지 않는다.
- [ ] RAG 질문은 노래방 기기 상태를 변경하지 않는다.
- [ ] 기존 `/api/v1/commands`, `/api/v1/stt`, `/api/v1/voice_command`가 회귀 없이 동작한다.
- [ ] 데스크톱에서는 오른쪽 2열, 모바일에서는 아래 1열이다.
- [ ] Python·JavaScript 문법 검사와 pytest가 통과한다.
- [ ] README만 보고 모델 설치와 실행을 재현할 수 있다.

---

## 5. 위험과 대응

### Ollama 임베딩 모델 미설치

- 증상: 첫 RAG 질문에서 연결 또는 모델 오류
- 대응: 시작 시 상태 확인, 503 응답, UI에 설치 안내

### 검색 점수 임계값 의미 차이

- Chroma API에 따라 similarity score 또는 distance 의미가 다를 수 있다.
- 실제 API 반환값을 확인하고 테스트에 점수 방향을 고정한다.
- 임계값을 근거 없이 임의 조정하지 않고 정답·거부 질문 세트로 검증한다.

### 문서가 짧아 검색이 과도하게 일치

- 기능별 제목을 유지하고 chunk 크기를 작게 시작한다.
- 무관 질문 거부 테스트를 반드시 통과시킨다.

### 초기 로딩 지연

- 서버 시작 시 Whisper와 벡터 인덱스를 모두 로드하면 지연이 커진다.
- Whisper와 RAG를 각각 첫 사용 시 로드하고 준비 상태를 UI에 표시한다.

### 공식 매뉴얼 오인

- 지식 문서를 `프로젝트 사용 안내`로 표시한다.
- 실제 제조사 문서를 사용하려면 출처와 사용 허가를 확인한 뒤 별도 교체한다.

---

## 6. 계획상 선택한 기본값

- RAG 패널 위치: 기존 기기 바깥 오른쪽
- 질문 범위: 사용법과 데모 문제 해결
- 벡터 저장소: Chroma
- 임베딩: Ollama `bge-m3`
- 생성 모델: 기존 `qwen3:1.7b`
- 검색 문서: 프로젝트 내부 Markdown
- 답변 언어: 한국어
- 기기 상태 변경: 금지
- 출처 표시: 필수
- 대화 이력: 초기 버전에서는 미사용
