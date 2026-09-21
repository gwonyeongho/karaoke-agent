// ====== CONFIG ======
const API_BASE = "http://127.0.0.1:8000";
const ENDPOINT = "/api/v1/commands";

// ====== STATE ======
let originStatus = {
  brand: "TJ",
  model_name: "B80",
  music: { pitch: 0, tempo: 0, melody_volume: 50 },
  sound: {
    echo_level: 50,
    reverb_type: "Hall",
    mic_volume: 70,
    inst_volume: 60,
    voice_cancel: false,
  },
  queue: { reserved_songs: [], played_song: null },
  is_playing: false,
  remaining_time: 0,
  remaining_coins: 0,
  subtitle_lang: "KOR",
  background_video_theme: "Nature",
};

let lastResponse = null;

// ====== DOM HELPERS (null-safe) ======

const $ = (id) => document.getElementById(id);
const safeText = (el, txt) => {
  if (el) el.textContent = txt;
};
const safeValue = (el, val) => {
  if (el) el.value = val;
};

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}
function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}
function setPill(el, text, type) {
  if (!el) return;
  el.textContent = text;
  el.style.borderColor =
    type === "ok"
      ? "rgba(56,240,183,0.55)"
      : type === "bad"
        ? "rgba(255,92,122,0.55)"
        : "rgba(37,48,74,0.9)";
  el.style.color =
    type === "ok"
      ? "rgba(56,240,183,0.9)"
      : type === "bad"
        ? "rgba(255,92,122,0.9)"
        : "";
}

// ====== ELEMENTS ======
const connPill = $("connPill");
const btnPing = $("btnPing");
const btnReset = $("btnReset");
const btnApplyResponse = $("btnApplyResponse");
const btnCopyJson = $("btnCopyJson");
const btnApplyLocal = $("btnApplyLocal");

const originJson = $("originJson");

const statusHint = $("statusHint");
const cmdInput = $("cmdInput");
const btnSend = $("btnSend");
const resultJson = $("resultJson");
const changedPill = $("changedPill");

const songNumber = $("songNumber");
const btnPlay = $("btnPlay");
const btnReserve = $("btnReserve");
const btnPriority = $("btnPriority");
const btnCancelReserve = $("btnCancelReserve");
const btnCancelPriority = $("btnCancelPriority");

// hidden sliders (dev panel)
const pitch = $("pitch");
const tempo = $("tempo");
const melodyHidden = $("melody"); // (숨김 슬라이더) - 있어도 되고 없어도 됨
const reverb = $("reverb");

// bottom controls (현재 HTML에 있음)
const mic = $("mic");
const inst = $("inst");
const echo = $("echo");

// (회전 다이얼용 melodyDial이 있으면 우선 사용)
const melodyDial = $("melodyDial"); // 없으면 null

// labels (center screen / knobs)
const pitchVal = $("pitchVal");
const tempoVal = $("tempoVal");
const melVal = $("melVal");

// 하단 노브 숫자 표시(HTML에 있음)
const micVal = $("micVal");
const instVal = $("instVal");
const echoVal = $("echoVal");
// melody 다이얼 숫자 표시(있으면 사용)
const melodyDialVal = $("melodyDialVal");

// meters
const mEcho = $("mEcho");
const mMic = $("mMic");
const mInst = $("mInst");
const mMelody = $("mMelody");
const nEchoVal = $("mEchoVal");
const mMicVal = $("mMicVal");
const mInstVal = $("mInstVal");
const mMelodyVal = $("mMelodyVal");

// screen header
const vBrand = $("vBrand");
const vModel = $("vModel");
const vPlay = $("vPlay");
const vTime = $("vTime");
const vCoins = $("vCoins");

// chips
const chipLang = $("chipLang");
const chipMR = $("chipMR");
const chipReverb = $("chipReverb");

// queue
const qReserved = $("qReserved");

// ====== RENDER ======
function renderMeters(status) {
  const echoP = clamp(status.sound?.echo_level ?? 0, 0, 100);
  const micP = clamp(status.sound?.mic_volume ?? 0, 0, 100);
  const instP = clamp(status.sound?.inst_volume ?? 0, 0, 100);
  const melP = clamp(status.music?.melody_volume ?? 0, 0, 100);

  if (mEcho) mEcho.style.width = echoP + "%";
  if (mMic) mMic.style.width = micP + "%";
  if (mInst) mInst.style.width = instP + "%";
  if (mMelody) mMelody.style.width = melP + "%";

  safeText(mEchoVal, String(echoP));
  safeText(mMicVal, String(micP));
  safeText(mInstVal, String(instP));
  safeText(mMelodyVal, String(melP));
}

function renderQueue(status) {
  const r = status.queue?.reserved_songs ?? [];
  safeText(qReserved, r.length ? r.join(", ") : "(비어있음)");
}

function renderHeader(status) {
  safeText(vBrand, status.brand ?? "TJ");
  safeText(vModel, status.model_name ?? "B80");

  safeText(
    vPlay,
    status.is_playing ? `PLAYING ${status.queue?.played_song}` : "STOP",
  );

  safeText(vTime, `${status.remaining_time ?? 0} min`);
  safeText(vCoins, String(status.remaining_coins ?? 0));
}

function renderChips(status) {
  if (chipLang) chipLang.textContent = String(status.subtitle_lang ?? "KOR");

  if (chipMR) {
    const isOn = !!status.sound?.voice_cancel;

    chipMR.textContent = `MR: ${isOn ? "ON" : "OFF"}`;

    chipMR.classList.remove("mr-on", "mr-off");
    chipMR.classList.add(isOn ? "mr-on" : "mr-off");
  }

  if (chipReverb)
    chipReverb.textContent = String(status.sound?.reverb_type ?? "Hall");
}

function syncControlsFromOrigin(status) {
  // dev hidden sliders
  safeValue(pitch, status.music?.pitch ?? 0);
  safeValue(tempo, status.music?.tempo ?? 0);
  safeValue(melodyHidden, status.music?.melody_volume ?? 0);
  safeValue(reverb, status.sound?.reverb_type ?? "Hall");

  // bottom ranges (값 저장용)
  safeValue(mic, status.sound?.mic_volume ?? 0);
  safeValue(inst, status.sound?.inst_volume ?? 0);
  safeValue(echo, status.sound?.echo_level ?? 0);

  // melodyDial이 있으면 거기로 동기화
  if (melodyDial) safeValue(melodyDial, status.music?.melody_volume ?? 0);

  // screen labels
  safeText(pitchVal, String(status.music?.pitch ?? 0));
  safeText(tempoVal, String(status.music?.tempo ?? 0));
  safeText(melVal, String(status.music?.melody_volume ?? 0));

  // knob labels
  safeText(micVal, String(status.sound?.mic_volume ?? 0));
  safeText(instVal, String(status.sound?.inst_volume ?? 0));
  safeText(echoVal, String(status.sound?.echo_level ?? 0));
  if (melodyDialVal)
    safeText(melodyDialVal, String(status.music?.melody_volume ?? 0));
}

function renderJson(status) {
  if (originJson) originJson.value = pretty(status);
}

function renderDisplay(status) {
  renderHeader(status);
  renderChips(status);
  renderMeters(status);
  renderQueue(status);
  syncControlsFromOrigin(status);
  renderJson(status);
  // 회전 다이얼이 있으면 포인터도 값에 맞춰 갱신
  updateAllDialVisuals();
}

// ====== LOCAL UPDATES (노브/슬라이더 움직일 때 즉시 originStatus 반영) ======
function setOriginByControl(id, valueNum) {
  // 이 함수는 "값만" 반영하고, 필요한 화면만 업데이트
  if (id === "echo") originStatus.sound.echo_level = valueNum;
  else if (id === "mic") originStatus.sound.mic_volume = valueNum;
  else if (id === "inst") originStatus.sound.inst_volume = valueNum;
  else if (id === "melodyDial" || id === "melody")
    originStatus.music.melody_volume = valueNum;
  else if (id === "pitch") originStatus.music.pitch = valueNum;
  else if (id === "tempo") originStatus.music.tempo = valueNum;

  // 빠른 부분 업데이트
  renderMeters(originStatus);
  renderChips(originStatus);
  syncControlsFromOrigin(originStatus);
  renderJson(originStatus);
}

function applyLocalControlsToOrigin() {
  originStatus = {
    ...originStatus,
    music: {
      ...originStatus.music,
      pitch: Number(pitch?.value ?? originStatus.music.pitch),
      tempo: Number(tempo?.value ?? originStatus.music.tempo),
      melody_volume: Number(
        melodyDial?.value ??
          melodyHidden?.value ??
          originStatus.music.melody_volume,
      ),
    },
    sound: {
      ...originStatus.sound,
      mic_volume: Number(mic?.value ?? originStatus.sound.mic_volume),
      inst_volume: Number(inst?.value ?? originStatus.sound.inst_volume),
      echo_level: Number(echo?.value ?? originStatus.sound.echo_level),
      reverb_type: reverb?.value ?? originStatus.sound.reverb_type,
    },
  };

  renderDisplay(originStatus);
  if (statusHint)
    statusHint.textContent = "로컬 컨트롤 값을 origin_status에 반영했습니다.";
}

// ====== NETWORK ======
async function ping() {
  try {
    const res = await fetch(API_BASE + "/docs", { method: "GET" });
    if (!res.ok) throw new Error("not ok");
    setPill(connPill, "API: CONNECTED", "ok");
  } catch {
    setPill(connPill, "API: DISCONNECTED", "bad");
  }
}

async function sendCommand(commandText) {
  if (!commandText || !commandText.trim()) return;

  if (statusHint) statusHint.textContent = "전송 중...";
  if (btnSend) btnSend.disabled = true;

  const payload = {
    command_text: commandText.trim(),
    origin_status: originStatus,
  };

  try {
    const res = await fetch(API_BASE + ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const t = await res.text();
      throw new Error(`HTTP ${res.status}\n${t}`);
    }

    const data = await res.json();
    lastResponse = data;

    if (resultJson) resultJson.value = pretty(data);

    if (data.changed === true) {
      setPill(changedPill, "changed: true", "ok");

      const { changed, reason, ...statusOnly } = data;
      originStatus = statusOnly;
      renderDisplay(originStatus);

      if (statusHint)
        statusHint.textContent = data.reason || "LLM 응답을 즉시 반영했습니다.";
      if (btnApplyResponse) btnApplyResponse.disabled = true;
    } else {
      setPill(changedPill, "changed: false", "bad");
      if (statusHint)
        statusHint.textContent = data.reason || "변경되지 않았습니다.";
      if (btnApplyResponse) btnApplyResponse.disabled = true;
    }
  } catch (err) {
    if (resultJson) resultJson.value = String(err);
    setPill(changedPill, "changed: -", "");
    if (statusHint)
      statusHint.textContent = "에러: API 연결/응답을 확인하세요.";
  } finally {
    if (btnSend) btnSend.disabled = false;
  }
}

function applyResponseToOrigin() {
  if (!lastResponse) return;
  const { changed, reason, ...statusOnly } = lastResponse;

  if (changed !== true) {
    if (statusHint)
      statusHint.textContent = "changed=false 응답은 반영하지 않습니다.";
    return;
  }

  originStatus = statusOnly;
  renderDisplay(originStatus);
  if (statusHint)
    statusHint.textContent = "응답을 origin_status에 반영했습니다.";
  if (btnApplyResponse) btnApplyResponse.disabled = true;
}

function resetOrigin() {
  originStatus = {
    brand: "TJ",
    model_name: "B80",
    music: { pitch: 0, tempo: 0, melody_volume: 50 },
    sound: {
      echo_level: 50,
      reverb_type: "Hall",
      mic_volume: 70,
      inst_volume: 60,
      voice_cancel: false,
    },
    queue: { reserved_songs: [], played_song: null },
    is_playing: false,
    remaining_time: 0,
    remaining_coins: 0,
    subtitle_lang: "KOR",
    background_video_theme: "Nature",
  };

  lastResponse = null;
  if (resultJson) resultJson.value = "";
  setPill(changedPill, "changed: -", "");
  if (btnApplyResponse) btnApplyResponse.disabled = true;
  if (statusHint) statusHint.textContent = "초기 상태로 리셋했습니다.";
  renderDisplay(originStatus);
}

async function copyJson() {
  if (!originJson) return;
  try {
    await navigator.clipboard.writeText(originJson.value);
    if (statusHint) statusHint.textContent = "origin_status JSON 복사 완료!";
  } catch {
    if (statusHint)
      statusHint.textContent = "복사 실패: 브라우저 권한을 확인하세요.";
  }
}

// ====== BUTTON BINDINGS ======
function bindDataCmdButtons() {
  document.querySelectorAll("[data-cmd]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cmd = btn.getAttribute("data-cmd");
      if (cmdInput) cmdInput.value = cmd;
      sendCommand(cmd);
    });
  });
}

function getSongNo() {
  if (!songNumber) return null;
  const raw = String(songNumber.value ?? "").trim();
  if (raw === "") return null;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) return null;
  songNumber.value = "";
  return Math.trunc(n);
}

function bindReservationButtons() {
  if (btnPlay)
    btnPlay.addEventListener("click", () => {
      const n = getSongNo();
      const cmd = (n === null ? `예약된 곡 시작` : `재생 시작 ${n}`);
      if (cmdInput) cmdInput.value = cmd;
      sendCommand(cmd);
    });
  if (btnReserve)
    btnReserve.addEventListener("click", () => {
      const n = getSongNo();
      if (n === null) {
        if (statusHint)
          statusHint.textContent = "곡 번호를 올바르게 입력하세요.";
        return;
      }
      const cmd = `예약 ${n}`;
      if (cmdInput) cmdInput.value = cmd;
      sendCommand(cmd);
    });

  if (btnPriority)
    btnPriority.addEventListener("click", () => {
      const n = getSongNo();
      if (n === null) {
        if (statusHint)
          statusHint.textContent = "곡 번호를 올바르게 입력하세요.";
        return;
      }
      const cmd = `우선예약 ${n}`;
      if (cmdInput) cmdInput.value = cmd;
      sendCommand(cmd);
    });

  if (btnCancelReserve)
    btnCancelReserve.addEventListener("click", () => {
      const n = getSongNo();
      if (n === null) {
        if (statusHint)
          statusHint.textContent = "곡 번호를 올바르게 입력하세요.";
        return;
      }
      const cmd = `예약 취소 ${n}`;
      if (cmdInput) cmdInput.value = cmd;
      sendCommand(cmd);
    });

  if (btnCancelPriority)
    btnCancelPriority.addEventListener("click", () => {
      const n = getSongNo();
      if (n === null) {
        if (statusHint)
          statusHint.textContent = "곡 번호를 올바르게 입력하세요.";
        return;
      }
      const cmd = `우선예약 취소 ${n}`;
      if (cmdInput) cmdInput.value = cmd;
      sendCommand(cmd);
    });
}

function bindCommandInput() {
  if (btnSend)
    btnSend.addEventListener("click", () => sendCommand(cmdInput?.value ?? ""));
  if (cmdInput)
    cmdInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendCommand(cmdInput.value);
    });
}

function bindLocalControlsLive() {
  // ✅ 노브/슬라이더를 움직이면 originStatus에 즉시 반영 (회전 다이얼도 input을 발생시키므로 여기서 함께 처리됨)
  const onInput = (el) => {
    if (!el) return;
    el.addEventListener("input", () => {
      const id = el.id;
      const n = Number(el.value);
      if (!Number.isFinite(n)) return;
      setOriginByControl(id, n);
    });
  };

  onInput(echo);
  onInput(mic);
  onInput(inst);
  onInput(melodyDial); // 있으면
  onInput(melodyHidden); // 없으면(숨김 슬라이더라도)

  onInput(pitch);
  onInput(tempo);

  if (btnApplyLocal)
    btnApplyLocal.addEventListener("click", applyLocalControlsToOrigin);
}

// ====== VOICE RECORD + UPLOAD (Server STT) ======
let mediaRecorder = null;
let chunks = [];

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];

    // webm이 가장 흔함
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/webm" });

      try {
        // ✅ STT 완료 즉시 콘솔 출력
        const stt = await uploadToSTT(blob);
        console.log("🎤 STT:", stt.text);

        // 입력칸에도 바로 보여주기
        if (cmdInput) cmdInput.value = stt.text ?? "";

        // STT 결과로 LLM 실행
        if (stt.text && stt.text.trim()) {
          await sendCommand(stt.text.trim());
        } else {
          if (statusHint)
            statusHint.textContent =
              "음성 인식 결과가 비어있어요(무음/인식 실패).";
        }
      } catch (e) {
        console.error("STT error:", e);
        if (statusHint)
          statusHint.textContent = "STT 실패: 서버 /ffmpeg /모델을 확인하세요.";
      } finally {
        stream.getTracks().forEach((t) => t.stop());
      }
    };

    mediaRecorder.start();
    if (statusHint) statusHint.textContent = "🎙️ 녹음 중... (Stop 누르면 전송)";
  } catch (e) {
    if (statusHint) statusHint.textContent = "마이크 권한이 필요합니다.";
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    if (statusHint) statusHint.textContent = "업로드/인식 중...";
  }
}

async function uploadToSTT(blob) {
  const fd = new FormData();
  fd.append("audio", blob, "voice.webm");
  fd.append("language_hint", "ko");

  const res = await fetch(API_BASE + "/api/v1/stt", {
    method: "POST",
    body: fd,
  });

  if (!res.ok) throw new Error(await res.text());
  return await res.json(); // {text, language, duration_sec}
}

$("btnMic")?.addEventListener("click", startRecording);
$("btnStopMic")?.addEventListener("click", stopRecording);

// ====== ROTARY DIAL (선택: HTML에 .dial이 있을 때만 동작) ======
const DIAL_MIN_ANGLE = -135;
const DIAL_MAX_ANGLE = 135;

function valueToAngle(value, min, max) {
  const t = (value - min) / (max - min);
  return DIAL_MIN_ANGLE + t * (DIAL_MAX_ANGLE - DIAL_MIN_ANGLE);
}

function angleToValue(angle, min, max, step = 1) {
  const a = clamp(angle, DIAL_MIN_ANGLE, DIAL_MAX_ANGLE);
  const t = (a - DIAL_MIN_ANGLE) / (DIAL_MAX_ANGLE - DIAL_MIN_ANGLE);
  const raw = min + t * (max - min);
  const snapped = Math.round(raw / step) * step;
  return clamp(snapped, min, max);
}

function getAngleFromPointer(e, centerX, centerY) {
  const x = e.clientX - centerX;
  const y = e.clientY - centerY;

  let deg = Math.atan2(y, x) * (180 / Math.PI);
  deg = deg + 90;

  if (deg > 180) deg -= 360;
  if (deg < -180) deg += 360;

  return deg;
}

function updateDialVisual(dialEl, inputEl) {
  const pointer = dialEl?.querySelector?.(".dial-pointer");
  if (!pointer || !inputEl) return;

  const min = Number(inputEl.min ?? 0);
  const max = Number(inputEl.max ?? 100);
  const val = Number(inputEl.value ?? 0);

  const angle = valueToAngle(val, min, max);
  pointer.style.transform = `translate(-50%, -90%) rotate(${angle}deg)`;
}

function updateAllDialVisuals() {
  document.querySelectorAll(".dial[data-input]").forEach((dialEl) => {
    const inputId = dialEl.getAttribute("data-input");
    const inputEl = inputId ? $(inputId) : null;
    if (inputEl) updateDialVisual(dialEl, inputEl);
  });
}

function initRotaryDials() {
  const dials = document.querySelectorAll(".dial[data-input]");
  if (!dials.length) return; // HTML에 다이얼이 없으면 그냥 스킵

  dials.forEach((dialEl) => {
    const inputId = dialEl.getAttribute("data-input");
    const valueId = dialEl.getAttribute("data-value");
    const inputEl = inputId ? $(inputId) : null;
    const valueEl = valueId ? $(valueId) : null;
    if (!inputEl) return;

    updateDialVisual(dialEl, inputEl);

    let dragging = false;

    const onMove = (ev) => {
      if (!dragging) return;
      const rect = dialEl.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;

      const angle = getAngleFromPointer(ev, cx, cy);
      const min = Number(inputEl.min ?? 0);
      const max = Number(inputEl.max ?? 100);
      const step = Number(inputEl.step ?? 1);

      const newVal = angleToValue(angle, min, max, step);

      inputEl.value = String(newVal);
      if (valueEl) valueEl.textContent = String(newVal);

      updateDialVisual(dialEl, inputEl);

      // ✅ 기존 input 핸들러(즉시 origin 반영) 트리거
      inputEl.dispatchEvent(new Event("input", { bubbles: true }));
    };

    const onUp = () => {
      if (!dragging) return;
      dragging = false;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };

    dialEl.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      dragging = true;
      dialEl.setPointerCapture?.(ev.pointerId);
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      onMove(ev);
    });

    inputEl.addEventListener("input", () => updateDialVisual(dialEl, inputEl));
  });
}

// ====== RAG ASSISTANT (isolated from device state) ======
const ragForm = $("ragForm");
const ragQuestion = $("ragQuestion");
const btnRagAsk = $("btnRagAsk");
const btnRagDocuments = $("btnRagDocuments");
const ragStatus = $("ragStatus");
const ragAnswer = $("ragAnswer");
const ragSourceList = $("ragSourceList");
const ragDocumentDialog = $("ragDocumentDialog");
const ragDocumentList = $("ragDocumentList");
const ragDocumentName = $("ragDocumentName");
const ragDocumentBody = $("ragDocumentBody");
let ragDocuments = [];

async function fetchJson(path, options) {
  const response = await fetch(API_BASE + path, options);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { const data = await response.json(); message = data.detail || message; } catch {}
    throw new Error(message);
  }
  return response.json();
}

async function loadRagDocuments() {
  ragDocuments = await fetchJson("/api/v1/rag/documents");
  ragDocumentList.replaceChildren();
  if (!ragDocuments.length) {
    const empty = document.createElement("p"); empty.className = "rag-empty";
    empty.textContent = "등록된 검색 자료가 없습니다."; ragDocumentList.append(empty); return;
  }
  ragDocuments.forEach((doc) => {
    const button = document.createElement("button"); button.type = "button";
    button.className = "rag-doc-button"; button.dataset.documentId = doc.id;
    const title = document.createElement("strong"); title.textContent = doc.title;
    const filename = document.createElement("small"); filename.textContent = doc.filename;
    button.append(title, filename);
    button.addEventListener("click", () => openRagDocument(doc.id));
    ragDocumentList.append(button);
  });
}

async function openRagDocument(documentId) {
  const doc = await fetchJson(`/api/v1/rag/documents/${encodeURIComponent(documentId)}`);
  ragDocumentName.textContent = `${doc.title} · ${doc.filename}`;
  ragDocumentBody.textContent = doc.content;
}

async function showRagDocuments(preferredFilename = null) {
  try {
    if (!ragDocuments.length) await loadRagDocuments();
    ragDocumentDialog.showModal();
    const selected = preferredFilename
      ? ragDocuments.find((doc) => doc.filename === preferredFilename)
      : ragDocuments[0];
    if (selected) await openRagDocument(selected.id);
  } catch (error) {
    safeText(ragStatus, `검색 자료를 불러오지 못했습니다: ${error.message}`);
  }
}

function renderRagSources(sources) {
  ragSourceList.replaceChildren();
  if (!sources?.length) {
    const empty = document.createElement("p"); empty.className = "rag-empty";
    empty.textContent = "표시할 검색 근거가 없습니다."; ragSourceList.append(empty); return;
  }
  sources.forEach((source) => {
    const card = document.createElement("div"); card.className = "rag-source-card";
    const name = document.createElement("strong"); name.textContent = source.source;
    const excerpt = document.createElement("p"); excerpt.textContent = source.excerpt;
    const open = document.createElement("button"); open.type = "button"; open.className = "btn ghost";
    open.textContent = "원문 보기"; open.addEventListener("click", () => showRagDocuments(source.source));
    card.append(name, excerpt, open); ragSourceList.append(card);
  });
}

async function askRag(question) {
  const trimmed = question.trim();
  if (!trimmed) { safeText(ragStatus, "질문을 입력하세요."); return; }
  btnRagAsk.disabled = true; safeText(ragStatus, "관련 문서를 검색하고 있어요...");
  ragAnswer.textContent = ""; renderRagSources([]);
  try {
    const data = await fetchJson("/api/v1/rag/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: trimmed, top_k: 4 }),
    });
    ragAnswer.textContent = data.answer;
    safeText(ragStatus, data.grounded ? "검색 문서를 근거로 답했어요." : "관련 근거를 찾지 못했어요.");
    renderRagSources(data.sources);
  } catch (error) {
    safeText(ragStatus, `RAG를 사용할 수 없습니다: ${error.message}`);
  } finally { btnRagAsk.disabled = false; }
}

ragForm?.addEventListener("submit", (event) => { event.preventDefault(); askRag(ragQuestion.value); });
btnRagDocuments?.addEventListener("click", () => showRagDocuments());
$("btnCloseRagDocuments")?.addEventListener("click", () => ragDocumentDialog.close());
ragDocumentDialog?.addEventListener("click", (event) => {
  if (event.target === ragDocumentDialog) ragDocumentDialog.close();
});

// ====== INIT ======
btnPing?.addEventListener("click", ping);
btnReset?.addEventListener("click", resetOrigin);
btnApplyResponse?.addEventListener("click", applyResponseToOrigin);
btnCopyJson?.addEventListener("click", copyJson);

bindDataCmdButtons();
bindReservationButtons();
bindCommandInput();
bindLocalControlsLive();
initRotaryDials();

renderDisplay(originStatus);
ping();
