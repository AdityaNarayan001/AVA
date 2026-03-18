# AVA — Real-Time Voice AI Pipeline

A real-time voice assistant pipeline for testing IoT voice AI stacks on your laptop before deploying to hardware.

**Pipeline:** Mic → TEN VAD → faster-whisper (STT) → LLM → kokoro-onnx (TTS) → Speaker

Supports **fully local** inference (Ollama) or **cloud LLM** APIs (OpenAI, Juspay Grid, OpenRouter, Together, etc.) — switchable via one line in `config.yaml`.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey)

---

## Features

- **Hands-free conversation** — continuous mic listening with VAD-based turn detection
- **Local or cloud LLM** — Ollama for offline, or any OpenAI-compatible API via config
- **Streaming pipeline** — LLM streams tokens → sentence-chunked TTS for low time-to-first-audio
- **Two UIs** — Cyberpunk terminal TUI (`terminal_app.py`) + Gradio web UI (`app.py`)
- **Real-time metrics** — per-turn latency breakdown (STT / LLM / TTS / E2E), session aggregates
- **Conversation memory** — sliding-window context management within the LLM's token budget
- **Fully configurable** — `config.yaml` controls every parameter; secrets in `.env` (git-safe)
- **One-click setup** — `setup.sh` handles everything: backend choice, API keys, deps, models
- **IoT-oriented** — small models chosen to represent edge deployment constraints

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AVA Voice Pipeline                           │
│                                                                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │   Mic   │───▶│   VAD   │───▶│   STT   │───▶│   LLM   │         │
│  │ 16kHz   │    │ TEN VAD │    │ faster- │    │smollm2  │         │
│  │ int16   │    │ 32ms    │    │ whisper │    │ :135m   │         │
│  │ mono    │    │ frames  │    │ (tiny)  │    │ Ollama  │         │
│  └─────────┘    └────┬────┘    └─────────┘    └────┬────┘         │
│                      │                              │               │
│                      │ speech                       │ text          │
│                      │ boundary                     │ response      │
│                      │ detection                    ▼               │
│                      │              ┌─────────┐  ┌─────────┐       │
│                      │              │ Speaker │◀─│   TTS   │       │
│                      │              │ 24kHz   │  │ kokoro  │       │
│                      │              │ float32 │  │  ONNX   │       │
│                      └──────────────┴─────────┘  └─────────┘       │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Gradio Web UI (:7860)                      │  │
│  │  Chat | Transcription | Per-Turn Metrics | Session Summary   │  │
│  │  VAD Meter | Settings | Memory Info | Export                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Threading Model

```
Audio Thread (sounddevice)     VAD Thread (daemon)             Processing Thread (daemon)
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ _mic_callback()      │    │ _vad_loop()          │    │ _processing_loop()   │
│                      │    │                      │    │                      │
│ Copy audio to ring   │ring│ Drain ring buffer    │ q  │ Dequeue utterance    │
│ buffer (minimal,     │───▶│ VAD.process() frames │───▶│  ├─ STT.transcribe() │
│ cffi-safe)           │    │ Buffer speech        │    │  ├─ LLM.generate()   │
│                      │    │ Detect silence       │    │  ├─ TTS.synthesize() │
│                      │    │  → enqueue utterance │    │  └─ sd.play() + wait │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

---

## Component Details

### 1. Voice Activity Detection (VAD)

| Property | Value |
|---|---|
| **Primary** | [TEN VAD](https://github.com/TEN-framework/ten-vad) — native C library via ctypes |
| **Fallback** | [Silero VAD](https://github.com/snakers4/silero-vad) — PyTorch-based |
| **Frame size** | 32ms (512 samples at 16kHz) |
| **Default threshold** | 0.5 (configurable 0.0–1.0) |
| **Input** | int16 numpy array, mono, 16kHz |
| **Output** | `VADResult(is_speech, confidence, backend)` |

**How it works:** Each 32ms audio frame from the mic is fed through the VAD. When speech is detected, frames are buffered. When silence exceeds the timeout (default 1000ms), the buffered audio is considered a complete utterance and queued for processing. Utterances shorter than `min_speech_duration` (default 500ms) are discarded as noise.

**Fallback logic:** On startup, `VADProcessor` tries to instantiate TEN VAD. If the native library isn't available, it falls back to Silero VAD (requires PyTorch). If both fail, the pipeline won't start.

| Setting | Default | Range | Effect |
|---|---|---|---|
| VAD Threshold | 0.5 | 0.0 – 1.0 | Higher = only loud/clear speech triggers |
| Silence Timeout | 1000ms | 300 – 3000ms | How long to wait after speech stops before processing |
| Min Speech Duration | 500ms | 100 – 2000ms | Reject utterances shorter than this |

---

### 2. Speech-to-Text (STT)

| Property | Value |
|---|---|
| **Engine** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend) |
| **Model** | `tiny` — 39M parameters, ~75MB |
| **Device** | CPU |
| **Compute type** | int8 (quantized) |
| **Beam size** | 1 (greedy decoding for speed) |
| **Language** | English (forced — skips language detection) |
| **Internal VAD** | Disabled (already handled upstream) |
| **Input** | int16 numpy array → converted to float32, written as temp WAV |
| **Output** | `STTResult(text, language, language_probability, duration_seconds, processing_time_ms, segments, realtime_factor)` |

**How it works:** The completed utterance (int16 at 16kHz) is normalized to float32 [-1, 1], written to a temporary WAV file, and passed to `faster-whisper`'s `model.transcribe()`. Per-segment details (timestamps, log probabilities) are captured for the transcription display.

---

### 3. Language Model (LLM)

| Property | Value |
|---|---|
| **Backend A** | [Ollama](https://ollama.com) — local inference |
| **Backend B** | OpenAI-compatible API — Juspay Grid, OpenRouter, Together, vLLM, etc. |
| **Default model (local)** | `smollm2:135m` — 135M parameters, ~270MB |
| **Default model (cloud)** | Configurable in `config.yaml` |
| **Streaming** | Yes — tokens stream to sentence chunker for overlapped TTS |
| **Selection** | `config.yaml` → `llm.backend`: `"ollama"` or `"openai_compatible"` |

**How it works:** The STT text is added to conversation memory. The memory module builds a message list (system prompt + sliding window of history) that fits within the token budget. This is sent to the active LLM backend. For Ollama, it uses the native Python client. For cloud APIs, it sends raw HTTP with SSE streaming and auto-detects OpenAI vs Anthropic response formats. Sentence boundaries are detected in the token stream and each sentence is immediately dispatched to TTS.

---

### 4. Text-to-Speech (TTS)

| Property | Value |
|---|---|
| **Engine** | [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) |
| **Model** | `kokoro-v1.0.onnx` (~310MB) + `voices-v1.0.bin` (~27MB) |
| **Output sample rate** | 24kHz |
| **Default voice** | `af_heart` |
| **Speed** | 1.0x |
| **Language** | en-us |
| **Runtime** | ONNX Runtime (CPU) |
| **System dependency** | `espeak-ng` (phonemization) |
| **Output** | `TTSResult(audio, processing_time_ms, audio_duration_seconds, realtime_factor, voice_name, sample_rate)` |

**Available voices:**

| Voice ID | Description |
|---|---|
| `af_heart` | American Female (default) |
| `af_bella` | American Female |
| `af_nicole` | American Female |
| `af_sarah` | American Female |
| `af_sky` | American Female |
| `am_adam` | American Male |
| `am_michael` | American Male |
| `bf_emma` | British Female |
| `bf_isabella` | British Female |
| `bm_george` | British Male |
| `bm_lewis` | British Male |

**How it works:** The LLM response text is passed to `kokoro_onnx.Kokoro`, which phonemizes via espeak-ng and synthesizes audio using the ONNX model. Output is a float32 numpy array at 24kHz, played through speakers via `sounddevice.play()`.

---

### 5. Conversation Memory

| Property | Value |
|---|---|
| **Strategy** | Sliding window (most recent turns prioritized) |
| **Default token budget** | 1500 tokens (of smollm2's ~2048 context) |
| **Token estimation** | `words × 1.3 + 4` per message (heuristic) |
| **System prompt** | Always included, never trimmed |
| **Stateless mode** | Only sends system prompt + current user message |

**How it works:** Messages are stored in full history. When building the context for the LLM, the memory module starts from the most recent message and works backward, adding messages until the token budget is exhausted. The system prompt is always included. Older messages are silently dropped. This ensures the LLM always has the most recent context while staying within its context window.

**Default system prompt:**
> You are AVA, a helpful and concise voice assistant running on a small device. Keep your responses short, natural, and conversational — typically 1-3 sentences. Speak in plain language. No markdown, no bullet points, no numbered lists. If you don't know something, say so briefly.

---

### 6. Metrics & Monitoring

**Per-turn metrics:**
| Metric | Description |
|---|---|
| STT Latency | Time to transcribe speech → text |
| LLM Latency | Time for model to generate response |
| TTS Latency | Time to synthesize speech |
| Total (E2E) | STT + LLM + TTS combined |
| Speech Duration | Length of user's utterance |
| STT Realtime Factor | How much faster than realtime STT runs |
| TTS Realtime Factor | How much faster than realtime TTS runs |
| LLM Tokens/sec | Generation speed |
| Context Tokens Used | How many tokens of the context window are filled |

**Session summary:** Aggregates across all turns — average, min, max, p95, and median for each latency stage. Also tracks total LLM tokens generated and session duration.

**Latency bar:** ASCII breakdown showing percentage of E2E time spent in each stage:
```
STT 15% | LLM 60% | TTS 25%
```

**Export:** Full session data + conversation history as JSON.

---

## Installation

### Prerequisites

- **Python 3.10+**
- **espeak-ng** — Required by kokoro-onnx for phonemization (setup.sh installs it)

### Quick Setup (Recommended)

The setup script is interactive — it asks you to choose your LLM backend and handles everything:

```bash
# Clone the repository
git clone https://github.com/AdityaNarayan001/AVA.git
cd AVA

# Run the setup script
chmod +x setup.sh
./setup.sh
```

**What setup.sh does:**
1. Asks you to choose **Ollama** (local) or **LiteLLM** (cloud API)
2. If Ollama → installs it, starts the server, pulls a model
3. If LiteLLM → asks for your API key, writes `.env`
4. Updates `config.yaml` with your choice
5. Installs espeak-ng if missing
6. Creates `.venv` and installs all Python dependencies
7. Downloads TTS models (~340 MB) from HuggingFace
8. Verifies everything works

> **Switching later:** Just edit `config.yaml` → `llm.backend` to `"ollama"` or `"openai_compatible"` and restart.

### Manual Setup

```bash
# 1. Install system dependencies
brew install espeak-ng           # macOS
# sudo apt install espeak-ng     # Linux

# 2. (Optional) Install Ollama for local LLM
brew install ollama && ollama pull smollm2:135m

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt
pip install -U git+https://github.com/TEN-framework/ten-vad.git

# 5. Download TTS models
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx https://huggingface.co/hexgrad/Kokoro-82M-v1.0-ONNX/resolve/main/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin  https://huggingface.co/hexgrad/Kokoro-82M-v1.0-ONNX/resolve/main/voices-v1.0.bin

# 6. Create .env for API keys (if using cloud LLM)
echo 'AVA_API_KEY=your-key-here' > .env
```

---

## Usage

### Start AVA

```bash
source .venv/bin/activate

# Terminal UI (recommended)
python terminal_app.py

# Or Gradio Web UI
python app.py              # opens http://localhost:7860
```

If using Ollama, make sure the server is running: `ollama serve &`

### How to Use

1. Click **"Start Listening"** — loads all models and opens the mic
2. **Just talk** — VAD detects your voice automatically
3. When you pause (~1s silence), AVA processes your speech:
   - Transcribes (STT) → Generates response (LLM) → Speaks it back (TTS)
4. The conversation, transcription, and metrics update live in the UI
5. Click **"Stop Listening"** when done

You can also type messages in the text box as an alternative input method.

### Run the Mic Diagnostic

If audio isn't working, run the diagnostic tool:

```bash
python test_mic.py
```

This tests mic capture, checks audio levels, and verifies VAD is detecting speech. If audio levels are near-zero, check your macOS microphone permissions (System Settings → Privacy & Security → Microphone).

---

## Project Structure

```
AVA/
├── config.yaml         # All pipeline parameters (backend, models, thresholds)
├── config.py           # Typed config loader with .env resolution
├── .env                # API keys (git-ignored, created by setup.sh)
├── pipeline.py         # Main orchestrator — mic → VAD → STT → LLM → TTS → speaker
├── vad.py              # Voice Activity Detection (TEN VAD + Silero fallback)
├── stt.py              # Speech-to-Text (faster-whisper)
├── llm.py              # LLM engine (Ollama local + OpenAI-compatible cloud)
├── tts.py              # Text-to-Speech (kokoro-onnx)
├── memory.py           # Conversation memory with sliding window
├── metrics.py          # Per-turn and session-level metrics
├── terminal_app.py     # Cyberpunk terminal UI (Rich)
├── app.py              # Gradio web UI
├── test_mic.py         # Mic + VAD diagnostic tool
├── setup.sh            # Interactive one-command setup
├── requirements.txt    # Python dependencies
└── models/             # Downloaded model files (git-ignored)
    ├── kokoro-v1.0.onnx    # TTS model (~310MB)
    └── voices-v1.0.bin     # TTS voice data (~27MB)
```

---

## Configuration Reference

All settings live in `config.yaml`. Secrets use `${VAR}` placeholders resolved from `.env`.

### LLM Backend

```yaml
llm:
  backend: "ollama"             # "ollama" or "openai_compatible"
  ollama:
    model: "smollm2:135m"       # Any Ollama model tag
    host: "http://localhost:11434"
  openai_compatible:
    base_url: "https://grid.ai.juspay.net/v1"
    api_key: "${AVA_API_KEY}"   # Resolved from .env
    model: "kimi-latest"
```

### Pipeline Settings

| Setting | Default | Range | Description |
|---|---|---|---|
| VAD Threshold | 0.5 | 0.0 – 1.0 | Speech detection sensitivity |
| Silence Timeout | 1.0s | 0.3 – 3.0s | Pause duration to end an utterance |
| Min Speech Duration | 0.5s | 0.1 – 2.0s | Minimum utterance length |
| TTS Voice | `af_heart` | 11 options | Kokoro voice selection |
| LLM Model | `smollm2:135m` | 3 options | Ollama model to use |
| Memory Enabled | Yes | On/Off | Toggle conversation context |
| Max Context Tokens | 1500 | 500 – 2000 | Token budget for memory |
| System Prompt | (see above) | Free text | Persona instructions for LLM |

---

## Key Constants

| Constant | Value | File |
|---|---|---|
| Mic sample rate | 16,000 Hz | `pipeline.py` |
| TTS sample rate | 24,000 Hz | `pipeline.py` |
| VAD frame duration | 32ms | `pipeline.py` |
| VAD frame samples | 512 | `pipeline.py` |
| UI poll interval | 500ms | `app.py` |
| Server port | 7860 | `app.py` |
| Token estimation ratio | 1.3 tokens/word | `memory.py` |

---

## Dependencies

| Component | Package | Purpose |
|---|---|---|
| Audio I/O | `sounddevice`, `soundfile`, `numpy`, `scipy` | Mic capture, audio playback |
| VAD (primary) | `ten-vad` | Native C VAD library |
| VAD (fallback) | `torch`, `torchaudio`, `silero-vad` | PyTorch-based VAD |
| STT | `faster-whisper` | Whisper tiny with CTranslate2 |
| LLM | `ollama` | Python client for Ollama |
| TTS | `kokoro-onnx`, `onnxruntime` | ONNX-based TTS |
| TTS (system) | `espeak-ng` | Phonemization backend |
| UI | `gradio` | Web interface |

---

## Approximate Resource Usage

| Component | RAM | Disk | Notes |
|---|---|---|---|
| TEN VAD | ~1MB | ~731KB | Native C, minimal |
| faster-whisper (tiny) | ~150MB | ~75MB | CTranslate2 int8 |
| smollm2:135m (Ollama) | ~300MB | ~270MB | Quantized by Ollama |
| kokoro-onnx | ~400MB | ~337MB | ONNX model + voices |
| Silero VAD (fallback) | ~100MB | ~50MB | PyTorch overhead |
| **Total (TEN VAD path)** | **~850MB** | **~680MB** | |
| **Total (Silero path)** | **~950MB** | **~730MB** | |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Port 7860 in use | `lsof -ti:7860 \| xargs kill -9` then retry |
| VAD meter stays at 0 | Run `python test_mic.py` — check mic permissions |
| No audio output | Check system volume and default output device |
| Ollama connection error | Make sure `ollama serve` is running (only needed for Ollama backend) |
| Cloud LLM timeout | Check API key in `.env` and `base_url` in `config.yaml` |
| espeak-ng not found | `brew install espeak-ng` (macOS) or `apt install espeak-ng` (Linux) |
| TEN VAD fails to load | Silero VAD auto-fallback activates; check terminal logs |
| Slow first response | Model loading on first use — subsequent turns are faster |

---

## License

MIT
