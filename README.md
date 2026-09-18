# Karaoke AI Agent

음성 또는 텍스트 명령을 해석해 노래방 기기의 상태를 제어하는 AI 에이전트 프로젝트입니다. 사용자의 명령과 현재 기기 상태를 Qwen에 전달하고 구조화된 결과를 검증한 뒤 웹 제어 화면에 반영합니다.

## 주요 기능

- faster-whisper 기반 음성 명령 인식
- Qwen3 1.7B 기반 자연어 명령 해석
- LangChain 구조화 출력과 Pydantic 데이터 검증
- 음정·템포·볼륨·에코·공간 음향·자막·예약곡 상태 제어
- 모호하거나 지원하지 않는 명령에 대한 기존 상태 유지
- FastAPI 백엔드와 HTML·CSS·JavaScript 웹 제어 화면 연동

## 성능 및 명령 처리 개선

초기 모델은 응답 시간이 길어 실제 조작 과정에서 지연이 발생했습니다. 실시간 사용성을 높이기 위해 최종 모델을 **Qwen3 1.7B**로 변경해 응답 속도를 확보했습니다.

모델 경량화 이후에는 사용자의 의도와 다른 명령이 반환되거나 지원 범위를 벗어난 값이 생성될 가능성을 보완했습니다.

- LangChain의 `with_structured_output`을 적용해 모델의 응답을 `KaraokeMachine` 스키마로 제한
- Pydantic으로 필드 타입과 음정·템포·볼륨 등의 허용 범위를 검증
- 현재 기기 상태를 명령과 함께 전달해 변경되지 않은 상태를 유지
- 모호하거나 지원하지 않는 요청은 기존 상태를 유지하도록 시스템 프롬프트에 규칙 명시
- `temperature=0`을 적용해 같은 조건에서 출력 변동을 줄임

이를 통해 경량 모델의 빠른 응답을 유지하면서 서비스에서 처리 가능한 형태의 결과만 상태 변경에 사용하도록 구성했습니다. LangChain은 모델 자체의 추론 성능을 높이는 용도가 아니라 구조화된 호출과 출력 연결에 사용했으며 Pydantic과 명령 규칙을 통해 실행 안정성을 보완했습니다.

## 기술 스택

- Python, FastAPI, Pydantic
- LangChain, ChatOllama, Ollama, Qwen3 1.7B
- faster-whisper
- HTML, CSS, Vanilla JavaScript

## 프로젝트 구조

```text
.
├── main.py           # FastAPI, STT, LLM 명령 처리 및 출력 검증
├── index.html        # 노래방 제어 화면
├── app.js            # 상태 관리와 API·음성 입력 연동
├── style.css         # 화면 스타일
└── requirements.txt  # Python 의존성
```

## 실행 방법

### 1. Ollama 및 모델 준비

Ollama를 설치한 뒤 Qwen3 1.7B 모델을 내려받습니다.

```bash
ollama pull qwen3:1.7b
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

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API 문서는 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.

### 4. 웹 화면 실행

새 터미널에서 다음 명령을 실행합니다.

```bash
python -m http.server 5500
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

## 명령 처리 흐름

```text
음성 녹음
→ faster-whisper 음성 인식
→ 현재 상태와 명령을 Qwen에 전달
→ LangChain 구조화 출력
→ Pydantic 타입·허용값·범위 검증
→ 검증된 상태를 웹 화면에 반영
```
