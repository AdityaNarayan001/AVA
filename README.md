# AVA — Real-Time Local Voice AI Pipeline

A fully local, real-time voice assistant pipeline for testing IoT voice AI stacks on your laptop before deploying to hardware.

**Pipeline:** Mic → TEN VAD → faster-whisper (STT) → smollm2:135m (LLM) → kokoro-onnx (TTS) → Speaker

No cloud APIs. No internet required at runtime. Just talk — AVA detects your voice, transcribes, thinks, and replies through your speakers automatically.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey)

---

## Features

- **Hands-free conversation** — continuous mic listening with VAD-based turn detection
- **Fully local** — every component runs on-device, no network calls
- **Real-time metrics** — per-turn latency breakdown (STT / LLM / TTS / E2E), session-level aggregates (avg, min, max, p95)
- **Live transcription** — see exactly what the STT model heard
- **Conversation memory** — sliding-window context management within the LLM's token budget
- **Configurable** — swap voices, models, VAD sensitivity, memory settings from the UI
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
Audio Thread (sounddevice)          Processing Thread (daemon)       Main Thread (Gradio)
┌──────────────────────┐           ┌──────────────────────┐        ┌──────────────────┐
│ _mic_callback()      │           │ _processing_loop()   │        │ gr.Timer (500ms) │
│                      │           │                      │        │                  │
│ For each 32ms frame: │  queue    │ Dequeue utterance     │ shared │ poll_ui_updates()│
│  ├─ VAD.process()    │─────────▶│  ├─ STT.transcribe() │──state─▶│  ├─ get_snapshot│
│  ├─ Update confidence│ (speech)  │  ├─ LLM.generate()   │ (lock) │  ├─ Update chat │
│  ├─ Buffer speech    │           │  ├─ TTS.synthesize()  │        │  ├─ Update VAD  │
│  └─ Detect silence   │           │  ├─ sd.play() + wait │        │  └─ Update stats│
│     → enqueue        │           │  └─ Update UI state   │        │                  │
└──────────────────────┘           └──────────────────────┘        └──────────────────┘
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
| **Runtime** | [Ollama](https://ollama.com) (local inference server) |
| **Default model** | `smollm2:135m` — 135M parameters, ~270MB |
| **Alternative models** | `smollm2:360m`, `smollm2:1.7b` (selectable in UI) |
| **Context window** | ~2048 tokens |
| **API** | `ollama.Client().chat(model, messages)` |
| **Token metrics** | Extracted from Ollama response: `eval_count`, `prompt_eval_count`, `eval_duration` |

**How it works:** The STT text is added to conversation memory. The memory module builds a message list (system prompt + sliding window of history) that fits within the token budget. This is sent to Ollama's chat API. The response text and token metrics are captured. If the Ollama call fails, the user message is rolled back from memory.

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
- **Ollama** — [Install Ollama](https://ollama.com/download)
- **espeak-ng** — Required by kokoro-onnx for phonemization

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ava-voice-pipeline.git
cd ava-voice-pipeline

# Run the setup script (installs everything)
chmod +x setup.sh
./setup.sh
```

### Manual Setup

```bash
# 1. Install system dependencies
brew install ollama espeak-ng   # macOS
# sudo apt install espeak-ng    # Linux

# 2. Pull the LLM model
ollama pull smollm2:135m

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt
pip install -U git+https://github.com/TEN-framework/ten-vad.git  # TEN VAD (optional — Silero fallback)

# 5. Download TTS models (first run does this automatically, or manually):
mkdir -p models
# kokoro-v1.0.onnx (~310MB) and voices-v1.0.bin (~27MB) are downloaded on first TTS use
```

### Verify Installation

```bash
source .venv/bin/activate
python -c "
import sounddevice, faster_whisper, ollama, kokoro_onnx, gradio
print('All core imports OK')
try:
    import ten_vad; print('TEN VAD: OK')
except: print('TEN VAD: not available (Silero fallback will be used)')
"
```

---

## Usage

### Start the App

```bash
# Make sure Ollama is running
ollama serve &   # if not already running

# Activate environment and launch
source .venv/bin/activate
python app.py
```

Open **http://localhost:7860** in your browser.

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
ava-voice-pipeline/
├── app.py              # Gradio web UI — real-time polling, controls, settings
├── pipeline.py         # Main orchestrator — mic, VAD, STT, LLM, TTS, playback
├── vad.py              # Voice Activity Detection (TEN VAD + Silero fallback)
├── stt.py              # Speech-to-Text (faster-whisper tiny)
├── llm.py              # LLM inference (Ollama + smollm2)
├── tts.py              # Text-to-Speech (kokoro-onnx)
├── memory.py           # Conversation memory with sliding window
├── metrics.py          # Per-turn and session-level metrics
├── test_mic.py         # Mic + VAD diagnostic tool
├── setup.sh            # One-command setup script
├── requirements.txt    # Python dependencies
└── models/             # Downloaded model files
    ├── kokoro-v1.0.onnx    # TTS model (~310MB)
    └── voices-v1.0.bin     # TTS voice data (~27MB)
```

---

## Configuration Reference

All settings are adjustable from the Gradio UI's Settings panel.

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
| Ollama connection error | Make sure `ollama serve` is running |
| espeak-ng not found | `brew install espeak-ng` (macOS) or `apt install espeak-ng` (Linux) |
| TEN VAD fails to load | Silero VAD auto-fallback activates; check terminal logs |
| Slow first response | Model loading on first use — subsequent turns are faster |

---

## License

MIT
