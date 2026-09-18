import os
import json
import tempfile
from typing import List, Literal, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# STT
from faster_whisper import WhisperModel


# ----------------------------
# Domain Models
# ----------------------------
class Song(BaseModel):
    id: int = Field(ge=1, description="곡 번호")
    title: str = Field(description="곡 제목")
    singer: str = Field(description="가수")


class MusicControl(BaseModel):
    pitch: int = Field(default=0, ge=-6, le=6, description="음정, 키, 피치")
    tempo: int = Field(default=0, ge=-5, le=5, description="템포, 박자, 속도")
    melody_volume: int = Field(default=50, ge=0, le=100, description="멜로디, 노래")


class SoundEffect(BaseModel):
    echo_level: int = Field(default=50, ge=0, le=100, description="에코")
    reverb_type: Literal["Room", "Hall", "Stage", "None"] = Field(
        default="Hall", description="공간 음향, 음향"
    )
    mic_volume: int = Field(
        default=70, ge=0, le=100, description="마이크 볼륨, 마이크, 마이크 크기"
    )
    inst_volume: int = Field(default=60, ge=0, le=100, description="반주, inst")
    voice_cancel: bool = Field(default=False, description="MR, 엠알")


class PlayQueue(BaseModel):
    reserved_songs: List[int] = Field(default_factory=list, description="예약된 곡")
    played_song: Optional[int] = Field(description="재생 중인 곡")


class KaraokeMachine(BaseModel):
    brand: str = Field(..., examples=["TJ"])
    model_name: str = Field(..., examples=["B80"])

    music: MusicControl = Field(default_factory=MusicControl)
    sound: SoundEffect = Field(default_factory=SoundEffect)
    queue: PlayQueue = Field(default_factory=PlayQueue)

    is_playing: bool = Field(default=False)
    remaining_time: int = Field(default=0, ge=0)
    remaining_coins: int = Field(default=0, ge=0)

    subtitle_lang: Literal["KOR", "ENG", "JPN", "CHN"] = Field(default="KOR")


# ----------------------------
# API Schemas
# ----------------------------
class CommandRequest(BaseModel):
    command_text: str
    origin_status: KaraokeMachine


class LLMResponse(KaraokeMachine):
    changed: bool
    reason: str = ""


class STTResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None


class VoiceCommandRequest(BaseModel):
    origin_status: KaraokeMachine
    # 필요하면 언어 지정 가능
    language_hint: Optional[str] = Field(default="ko", description="예: 'ko', 'en'")


# ----------------------------
# LLM
# ----------------------------
llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0,
)

SYSTEM_PROMPT = """
너는 사용자의 자연어 명령을 받아서 노래방 기계를 제어하는 에이전트다.
입력으로 현재 기기 상태(JSON)와 사용자 명령이 주어진다.

너의 출력은 '변경된 기기 상태'여야 한다.
하지만 아래 조건이면 '상태를 변경하지 말고' 입력 상태를 그대로 유지해야 한다.

[변경 금지 조건]
- 사용자의 요청이 이 스키마에 없는 기능을 요구함
- 필드 범위를 벗어남 (예: pitch 10, tempo 20, 볼륨 150 등)
- song 번호가 음수이거나 말이 안되는 값

규칙:
- 재생 시작: "재생 시작 (숫자)" => played_song에 저장하고 is_playing=True
- 예약된 곡 시작 => reserved_songs의 첫번째 요소를 빼와서 그 곡을 played_song에 저장하고 is_playing=True
- 예약: "예약 (숫자)" => reserved_songs의 뒤에 추가
- 우선예약/새치기: reserved_songs의 앞에 추가
- 예약 취소: reserved_songs에서 제거
- 우선예약 취소: priority_songs에서 제거
- MR 모드/보컬 제거 => voice_cancel=True
- MR 해제/보컬 살려 => voice_cancel=False
- 자막 영어 => subtitle_lang=ENG (KOR/ENG/JPN/CHN)
- 정지 => played_song을 None으로 변환하고 is_playing=False
- 키/피치:
  - "키 2", "피치 2" => music.pitch=2
  - "키 올려줘" => pitch +1 (단 -6~+6 클램프)
  - "키 내려줘" => pitch -1 (단 -6~+6 클램프)
"""


def _compute_changed(origin: KaraokeMachine, updated: KaraokeMachine) -> bool:
    return origin.model_dump() != updated.model_dump()


def run_llm_command(origin: KaraokeMachine, command_text: str) -> LLMResponse:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "현재 상태(JSON):\n"
            + json.dumps(origin.model_dump(), ensure_ascii=False, indent=2),
        },
        {"role": "user", "content": "사용자 명령:\n" + command_text},
        {"role": "user", "content": "위 규칙을 지켜서 '변경된 상태'만 출력해."},
    ]

    updated_status: KaraokeMachine = llm.with_structured_output(KaraokeMachine).invoke(
        messages
    )
    changed = _compute_changed(origin, updated_status)
    reason = (
        "OK" if changed else "명령이 모호/불가/범위초과로 판단되어 변경하지 않았습니다."
    )
    return LLMResponse(**updated_status.model_dump(), changed=changed, reason=reason)


# ----------------------------
# STT (faster-whisper)
# ----------------------------
# 모델 옵션:
# - "small" 추천(속도/품질 균형)
# - GPU 있으면 device="cuda", compute_type="float16"
# - CPU면 device="cpu", compute_type="int8" 추천
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

whisper_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE,
)


def transcribe_file(path: str, language_hint: Optional[str] = "ko") -> STTResponse:
    # language_hint: "ko" / "en" 등. None이면 자동감지
    segments, info = whisper_model.transcribe(
        path,
        language=language_hint if language_hint else None,
        vad_filter=True,  # 무음 제거(권장)
        vad_parameters={"min_silence_duration_ms": 300},
    )

    text = "".join(seg.text for seg in segments).strip()
    return STTResponse(
        text=text,
        language=getattr(info, "language", None),
        duration_sec=getattr(info, "duration", None),
    )


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"ok": True}


@app.post("/api/v1/commands", response_model=LLMResponse)
def commands(command_request: CommandRequest):
    return run_llm_command(command_request.origin_status, command_request.command_text)


@app.post("/api/v1/stt", response_model=STTResponse)
async def stt(audio: UploadFile = File(...), language_hint: Optional[str] = "ko"):
    # 간단한 파일 타입 체크(너무 엄격하면 브라우저 webm이 막힐 수 있어 느슨히)
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file")

    # 임시 파일로 저장
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await audio.read()
        # 너무 큰 파일 방어(예: 15MB)
        if len(data) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        tmp.write(data)
        tmp_path = tmp.name

    try:
        res = transcribe_file(tmp_path, language_hint=language_hint)
        if not res.text:
            # 말이 없거나 인식 실패
            return STTResponse(
                text="", language=res.language, duration_sec=res.duration_sec
            )
        return res
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/api/v1/voice_command", response_model=LLMResponse)
async def voice_command(
    req_json: str = File(..., description="VoiceCommandRequest를 JSON string으로"),
    audio: UploadFile = File(...),
):
    """
    multipart/form-data로 받기:
      - req_json: {"origin_status": {...}, "language_hint":"ko"} 문자열
      - audio: 녹음 파일(webm/ogg/wav 등)
    """

    # 1) JSON 파싱
    try:
        req = VoiceCommandRequest.model_validate_json(req_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad req_json: {e}")

    # 2) 오디오 저장
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await audio.read()
        if len(data) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # 3) STT
        stt_res = transcribe_file(tmp_path, language_hint=req.language_hint)

        if not stt_res.text:
            # 음성 인식 실패 -> 변경하지 않음
            return LLMResponse(
                **req.origin_status.model_dump(),
                changed=False,
                reason="음성 인식 실패(무음/인식불가)",
            )

        # 4) LLM 제어
        llm_res = run_llm_command(req.origin_status, stt_res.text)

        # reason에 STT 텍스트를 붙이고 싶으면(선택)
        # llm_res.reason = f"{llm_res.reason} | STT: {stt_res.text}"

        return llm_res

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
