"""
Audio Pipeline for AVA.
Orchestrates: Mic → VAD → STT → LLM (with memory) → TTS → Speaker
Continuous listening with automatic speech detection and response.
"""

import numpy as np
import sounddevice as sd
import time
import threading
import queue
import collections
import logging
from typing import Optional, Callable, Tuple, List, Dict
from dataclasses import dataclass
from scipy.signal import resample_poly
from math import gcd

from config import get_config
from vad import VADProcessor, VADResult
from stt import STTEngine, STTResult
from llm import LLMEngine, LLMResult
from tts import TTSEngine, TTSResult
from memory import ConversationMemory, StatelessMemory, DEFAULT_SYSTEM_PROMPT
from metrics import PipelineMetrics, SessionMetrics

logger = logging.getLogger(__name__)

_cfg = get_config()

SAMPLE_RATE = _cfg.audio.input_sample_rate
TTS_SAMPLE_RATE = _cfg.audio.tts_sample_rate
VAD_FRAME_MS = _cfg.vad.frame_ms
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)

# Query native output device sample rate once at import
try:
    _OUTPUT_DEVICE_RATE = int(sd.query_devices(kind='output')['default_samplerate'])
except Exception:
    _OUTPUT_DEVICE_RATE = 48000

# Pre-compute resampling ratio (avoids GCD calc on every playback)
_RESAMPLE_UP = _OUTPUT_DEVICE_RATE // gcd(TTS_SAMPLE_RATE, _OUTPUT_DEVICE_RATE)
_RESAMPLE_DOWN = TTS_SAMPLE_RATE // gcd(TTS_SAMPLE_RATE, _OUTPUT_DEVICE_RATE)


def _safe_play(audio: np.ndarray, samplerate: int):
    """Play audio, resampling to the device's native rate if needed."""
    if samplerate != _OUTPUT_DEVICE_RATE:
        audio = resample_poly(audio, _RESAMPLE_UP, _RESAMPLE_DOWN).astype(np.float32)
        samplerate = _OUTPUT_DEVICE_RATE
    sd.play(audio, samplerate=samplerate)
    sd.wait()

# Silence detection defaults (from config)
DEFAULT_SILENCE_TIMEOUT_MS = _cfg.vad.silence_timeout_ms
DEFAULT_MIN_SPEECH_MS = _cfg.vad.min_speech_ms
DEFAULT_VAD_THRESHOLD = _cfg.vad.threshold


@dataclass
class PipelineOutput:
    """Result from processing a single utterance through the full pipeline."""
    metrics: PipelineMetrics
    stt_result: Optional[STTResult] = None
    llm_result: Optional[LLMResult] = None
    tts_result: Optional[TTSResult] = None
    error: Optional[str] = None


class VoicePipeline:
    """
    Main voice pipeline orchestrator.
    Manages the full flow from raw audio to synthesized speech,
    with metrics collection at each stage.
    """

    def __init__(self):
        # Engine components (lazy-loaded)
        self.vad: Optional[VADProcessor] = None
        self.stt: Optional[STTEngine] = None
        self.llm: Optional[LLMEngine] = None
        self.tts: Optional[TTSEngine] = None

        # Memory
        cfg_mem = get_config().memory
        if cfg_mem.enabled:
            self.memory: ConversationMemory = ConversationMemory()
        else:
            self.memory: ConversationMemory = StatelessMemory()

        # Metrics
        self.session_metrics = SessionMetrics()

        # Audio buffering state
        self._speech_buffer: list = []
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_start_time = 0.0

        # Configuration
        self.silence_timeout_ms = DEFAULT_SILENCE_TIMEOUT_MS
        self.min_speech_ms = DEFAULT_MIN_SPEECH_MS
        self.vad_threshold = DEFAULT_VAD_THRESHOLD

        # Pipeline state
        self._running = False
        self._processing = False
        self._playing = False
        self._status = "idle"  # idle, listening, speech_detected, transcribing, thinking, speaking

        # Continuous listening
        self._mic_stream: Optional[sd.InputStream] = None
        self._process_thread: Optional[threading.Thread] = None
        self._vad_thread: Optional[threading.Thread] = None
        self._audio_ring: collections.deque = collections.deque(maxlen=2000)
        self._utterance_queue: queue.Queue = queue.Queue()

        # Shared UI state (read by Gradio timer, written by pipeline threads)
        self._ui_lock = threading.Lock()
        self._chat_history: List[Dict[str, str]] = []
        self._last_transcription_md: str = "*Waiting for speech...*"
        self._last_per_turn_md: str = "*Metrics will appear after the first turn...*"
        self._last_session_md: str = "No turns completed yet."
        self._last_memory_md: str = "**Memory:** No conversation yet"
        self._last_component_info: str = "*Models will load on first interaction...*"
        self._last_vad_confidence: float = 0.0
        self._pending_tts_audio: Optional[tuple] = None  # (sample_rate, np.ndarray) or None
        self._last_output: Optional[PipelineOutput] = None
        self._ui_update_counter: int = 0  # increments on each update so timer can detect changes

        # Callbacks
        self._on_status_change: Optional[Callable] = None
        self._on_vad_update: Optional[Callable] = None

        logger.info("VoicePipeline created")

    def initialize(self):
        """Load all models. Call once before processing."""
        from concurrent.futures import ThreadPoolExecutor

        logger.info("Initializing pipeline components...")
        t0 = time.perf_counter()

        # VAD loads instantly (C library)
        self._set_status("loading_vad")
        self.vad = VADProcessor(hop_size=VAD_FRAME_SAMPLES, threshold=self.vad_threshold)

        # LLM client is lightweight (just socket setup)
        self.llm = LLMEngine()

        # STT and TTS model loads are independent — run in parallel
        self._set_status("loading_models")
        self.stt = STTEngine()
        self.tts = TTSEngine()

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ava-load") as pool:
            stt_future = pool.submit(self.stt.load)
            tts_future = pool.submit(self.tts.load)
            stt_future.result()  # wait for both
            tts_future.result()

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Pipeline initialized in {total_ms:.0f}ms")
        self._set_status("idle")

    def _set_status(self, status: str):
        self._status = status
        if self._on_status_change:
            self._on_status_change(status)

    def feed_audio(self, audio_chunk: np.ndarray) -> Tuple[Optional[VADResult], Optional[np.ndarray]]:
        """
        Feed a chunk of audio from the microphone into the VAD.
        
        Args:
            audio_chunk: int16 numpy array at 16kHz
            
        Returns:
            Tuple of (VADResult or None, completed_speech_audio or None)
            completed_speech_audio is returned when an utterance is complete.
        """
        if self.vad is None:
            return None, None

        completed_audio = None

        # Process through VAD in frames
        for i in range(0, len(audio_chunk), VAD_FRAME_SAMPLES):
            frame = audio_chunk[i:i + VAD_FRAME_SAMPLES]
            if len(frame) < VAD_FRAME_SAMPLES:
                # Pad short frames
                frame = np.pad(frame, (0, VAD_FRAME_SAMPLES - len(frame)))

            vad_result = self.vad.process(frame)

            # Update VAD callback for live UI
            if self._on_vad_update:
                self._on_vad_update(vad_result)

            if vad_result.is_speech:
                if not self._is_speaking:
                    # Speech started
                    self._is_speaking = True
                    self._speech_start_time = time.perf_counter()
                    self._speech_buffer = []
                    self._set_status("speech_detected")
                    logger.debug("Speech started")

                self._speech_buffer.append(frame.copy())
                self._silence_frames = 0

            elif self._is_speaking:
                # Count silence frames after speech
                self._silence_frames += 1
                # Still buffer during silence gap (captures trailing audio)
                self._speech_buffer.append(frame.copy())

                silence_ms = self._silence_frames * VAD_FRAME_MS
                if silence_ms >= self.silence_timeout_ms:
                    # Speech ended — check minimum duration
                    speech_duration_ms = len(self._speech_buffer) * VAD_FRAME_MS
                    if speech_duration_ms >= self.min_speech_ms:
                        completed_audio = np.concatenate(self._speech_buffer)
                        logger.info(f"Utterance complete: {speech_duration_ms:.0f}ms")
                    else:
                        logger.debug(f"Rejected short utterance: {speech_duration_ms:.0f}ms")

                    # Reset state
                    self._is_speaking = False
                    self._speech_buffer = []
                    self._silence_frames = 0
                    self.vad.reset()
                    if completed_audio is None:
                        self._set_status("listening")

        last_vad = vad_result if len(audio_chunk) >= VAD_FRAME_SAMPLES else None
        return last_vad, completed_audio

    def process_utterance(
        self,
        speech_audio: np.ndarray,
        on_tts_chunk: Optional[Callable[[TTSResult], None]] = None,
    ) -> PipelineOutput:
        """
        Process a complete utterance through STT → LLM (streaming) → TTS (sentence-chunked).

        Uses the LLM streaming API to detect sentence boundaries and starts TTS
        synthesis on the first complete sentence while the LLM is still generating.
        If `on_tts_chunk` is provided, each synthesized sentence is passed to it
        immediately (typically for playback), enabling overlapped audio.

        Args:
            speech_audio: int16 numpy array at 16kHz containing the speech
            on_tts_chunk: Optional callback receiving each TTSResult chunk for playback

        Returns:
            PipelineOutput with aggregated results and metrics
        """
        self._processing = True
        pipeline_t0 = time.perf_counter()
        metrics = PipelineMetrics()
        metrics.speech_duration_ms = len(speech_audio) / SAMPLE_RATE * 1000
        metrics.vad_backend = self.vad.backend_name if self.vad else "unknown"

        output = PipelineOutput(metrics=metrics)

        try:
            # ---- STT ----
            self._set_status("transcribing")
            stt_result = self.stt.transcribe(speech_audio)
            output.stt_result = stt_result

            metrics.stt_latency_ms = stt_result.processing_time_ms
            metrics.stt_text = stt_result.text
            metrics.stt_confidence = stt_result.language_probability
            metrics.stt_language = stt_result.language
            metrics.stt_realtime_factor = stt_result.realtime_factor

            if not stt_result.text.strip():
                logger.info("STT returned empty text, skipping LLM/TTS")
                self._set_status("listening")
                self._processing = False
                return output

            # ---- LLM (streaming) → TTS (sentence-chunked) ----
            self._set_status("thinking")

            tts_chunks: list[TTSResult] = []
            total_tts_ms = 0.0
            first_audio_time: Optional[float] = None

            def _on_sentence(sentence: str):
                """Called by LLM streamer for each complete sentence."""
                nonlocal total_tts_ms, first_audio_time

                self._set_status("speaking")
                tts_result = self.tts.synthesize(sentence)
                tts_chunks.append(tts_result)
                total_tts_ms += tts_result.processing_time_ms

                if first_audio_time is None:
                    first_audio_time = time.perf_counter()

                if on_tts_chunk is not None:
                    on_tts_chunk(tts_result)

            llm_result = self.llm.generate_stream(
                self.memory, stt_result.text, on_sentence=_on_sentence,
            )
            output.llm_result = llm_result

            metrics.llm_latency_ms = llm_result.processing_time_ms
            metrics.llm_response_text = llm_result.text
            metrics.llm_tokens = llm_result.token_count
            metrics.llm_prompt_tokens = llm_result.prompt_token_count
            metrics.llm_tokens_per_sec = llm_result.tokens_per_second
            metrics.llm_model = llm_result.model_name
            metrics.context_tokens_used = llm_result.context_tokens_used
            metrics.turns_in_context = llm_result.turns_in_context

            # ---- Aggregate TTS results ----
            if tts_chunks:
                # Concatenate all audio chunks into one result for callers
                # that still read output.tts_result (e.g., Gradio UI state)
                all_audio = np.concatenate([c.audio for c in tts_chunks])
                total_audio_dur = sum(c.audio_duration_seconds for c in tts_chunks)
                agg_tts = TTSResult(
                    audio=all_audio,
                    processing_time_ms=total_tts_ms,
                    audio_duration_seconds=total_audio_dur,
                    realtime_factor=total_audio_dur / (total_tts_ms / 1000) if total_tts_ms > 0 else 0,
                    voice_name=tts_chunks[0].voice_name,
                    sample_rate=tts_chunks[0].sample_rate,
                    text_length=len(llm_result.text),
                )
                output.tts_result = agg_tts

                metrics.tts_latency_ms = total_tts_ms
                metrics.tts_audio_duration_ms = total_audio_dur * 1000
                metrics.tts_realtime_factor = agg_tts.realtime_factor
                metrics.tts_voice = agg_tts.voice_name
                metrics.tts_sentences = len(tts_chunks)

            # Time to first audio = time from pipeline start to first TTS chunk ready
            if first_audio_time is not None:
                metrics.time_to_first_audio_ms = (first_audio_time - pipeline_t0) * 1000

            # Compute total
            metrics.compute_total()

            # Record in session metrics
            self.session_metrics.add_turn(metrics)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            output.error = str(e)
            self._set_status("error")

        self._processing = False
        return output

    # ─── Continuous Listening ────────────────────────────────

    def start_listening(self):
        """Start continuous microphone listening + auto-processing loop."""
        if self._running:
            logger.warning("Already listening")
            return

        self.initialize()
        self._update_component_info()
        self._running = True
        self._set_status("listening")

        # Start the processing thread (picks utterances from queue)
        self._process_thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="ava-process"
        )
        self._process_thread.start()

        # Start the VAD thread (drains audio ring → runs VAD outside cffi context)
        self._vad_thread = threading.Thread(
            target=self._vad_loop, daemon=True, name="ava-vad"
        )
        self._vad_thread.start()

        # Start the mic stream — callback ONLY copies data into the ring buffer
        self._mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=VAD_FRAME_SAMPLES,
            callback=self._mic_callback,
        )
        self._mic_stream.start()
        logger.info("Continuous listening started")

    def stop_listening(self):
        """Stop the microphone and processing loop."""
        self._running = False

        # Stop mic
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

        # Drain the queue so the processing thread can exit
        while not self._utterance_queue.empty():
            try:
                self._utterance_queue.get_nowait()
            except queue.Empty:
                break

        # Signal the processing thread to stop
        self._utterance_queue.put(None)
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=5)
        self._process_thread = None

        # Wait for VAD thread
        if self._vad_thread and self._vad_thread.is_alive():
            self._vad_thread.join(timeout=3)
        self._vad_thread = None

        self.reset_audio_state()
        self._set_status("idle")
        logger.info("Listening stopped")

    def _mic_callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice (PortAudio cffi) for every audio block.
        MUST be minimal — only copy data to the ring buffer.
        Heavy work (VAD) lives in _vad_loop to avoid cffi callback crashes.
        """
        if status:
            # cannot use logger inside cffi; deferred ring-side logging is fine
            pass
        if self._running:
            self._audio_ring.append(indata[:, 0].copy())

    def _vad_loop(self):
        """
        Drains audio from the ring buffer and runs VAD processing
        in a pure-Python thread (safe from cffi callback restrictions).
        """
        logger.info("VAD processing thread started")
        while self._running:
            # Drain all available chunks (non-blocking)
            chunks = []
            try:
                while True:
                    chunks.append(self._audio_ring.popleft())
            except IndexError:
                pass  # deque empty

            if not chunks:
                time.sleep(0.005)  # ~5 ms idle poll
                continue

            for audio_chunk in chunks:
                try:
                    for i in range(0, len(audio_chunk), VAD_FRAME_SAMPLES):
                        frame = audio_chunk[i:i + VAD_FRAME_SAMPLES]
                        if len(frame) < VAD_FRAME_SAMPLES:
                            frame = np.pad(frame, (0, VAD_FRAME_SAMPLES - len(frame)))

                        vad_result = self.vad.process(frame)
                        self._last_vad_confidence = vad_result.confidence

                        # Skip speech buffering while processing or playing
                        if self._processing or self._playing:
                            continue

                        if vad_result.is_speech:
                            if not self._is_speaking:
                                self._is_speaking = True
                                self._speech_start_time = time.perf_counter()
                                self._speech_buffer = []
                                self._set_status("speech_detected")
                            self._speech_buffer.append(frame.copy())
                            self._silence_frames = 0

                        elif self._is_speaking:
                            self._silence_frames += 1
                            self._speech_buffer.append(frame.copy())

                            silence_ms = self._silence_frames * VAD_FRAME_MS
                            if silence_ms >= self.silence_timeout_ms:
                                speech_duration_ms = len(self._speech_buffer) * VAD_FRAME_MS
                                if speech_duration_ms >= self.min_speech_ms:
                                    completed = np.concatenate(self._speech_buffer)
                                    self._utterance_queue.put(completed)
                                    logger.info(f"Utterance queued: {speech_duration_ms:.0f}ms")
                                else:
                                    logger.debug(f"Rejected short utterance: {speech_duration_ms:.0f}ms")

                                self._is_speaking = False
                                self._speech_buffer = []
                                self._silence_frames = 0
                                try:
                                    self.vad.reset()
                                except Exception:
                                    pass
                                if not self._processing:
                                    self._set_status("listening")

                except Exception as e:
                    logger.error(f"VAD loop error: {e}", exc_info=True)

        logger.info("VAD processing thread stopped")

    def _processing_loop(self):
        """
        Background thread: picks completed utterances from queue,
        runs STT → LLM (streaming) → TTS (sentence-chunked), plays audio.
        """
        logger.info("Processing loop started")
        while self._running:
            try:
                speech_audio = self._utterance_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if speech_audio is None:
                break  # Shutdown signal

            def _play_chunk(tts_result: TTSResult):
                """Play each TTS sentence immediately as it's synthesized."""
                self._playing = True
                try:
                    _safe_play(tts_result.audio, tts_result.sample_rate)
                except Exception as e:
                    logger.error(f"Audio playback error: {e}")

            # Process with streaming — audio plays inside _play_chunk
            output = self.process_utterance(speech_audio, on_tts_chunk=_play_chunk)
            self._playing = False
            self._last_output = output

            # Update shared UI state
            self._update_ui_state(output)

            if self._running:
                self._set_status("listening")

        logger.info("Processing loop stopped")

    def _update_ui_state(self, output: PipelineOutput):
        """Write updated data into shared state for the Gradio timer to read."""
        metrics = output.metrics
        sm = self.session_metrics

        with self._ui_lock:
            # Chat history
            user_text = metrics.stt_text or ""
            assistant_text = metrics.llm_response_text or ""
            if user_text.strip():
                self._chat_history.append({"role": "user", "content": user_text})
                if assistant_text.strip():
                    self._chat_history.append({"role": "assistant", "content": assistant_text})

            # Transcription
            if output.stt_result:
                stt = output.stt_result
                self._last_transcription_md = (
                    f"### Raw Transcription\n\n"
                    f"> {stt.text}\n\n"
                    f"**Language:** {stt.language} ({stt.language_probability:.0%})\n\n"
                    f"**Audio Duration:** {stt.duration_seconds:.1f}s\n\n"
                    f"**STT Processing:** {stt.processing_time_ms:.0f}ms "
                    f"({stt.realtime_factor:.1f}x realtime)"
                )
                if stt.segments:
                    self._last_transcription_md += "\n\n**Segments:**\n"
                    for seg in stt.segments:
                        self._last_transcription_md += (
                            f"- [{seg.start:.1f}s-{seg.end:.1f}s] {seg.text} "
                            f"(logprob: {seg.avg_logprob:.2f})\n"
                        )

            # Per-turn metrics
            if sm.turns:
                self._last_per_turn_md = sm.format_per_turn_metrics(metrics)
                latency_bar = sm.format_latency_bar(metrics)
                if latency_bar:
                    self._last_per_turn_md += f"\n\n**Pipeline Breakdown:** {latency_bar}"

            if output.error:
                self._last_per_turn_md = f"⚠️ **Error:** {output.error}\n\n" + self._last_per_turn_md

            # Session summary
            self._last_session_md = sm.format_session_summary()

            # Memory info
            mem_info = self.memory.get_context_info()
            self._last_memory_md = (
                f"**Memory:** {mem_info['turns_in_context']}/{mem_info['total_turns']} turns in context | "
                f"{mem_info['context_tokens_used']}/{mem_info['max_context_tokens']} tokens "
                f"({mem_info['context_utilization']:.0%})"
            )

            # TTS audio for Gradio playback
            if output.tts_result and output.tts_result.audio is not None:
                self._pending_tts_audio = (output.tts_result.sample_rate, output.tts_result.audio)

            self._ui_update_counter += 1

    def _update_component_info(self):
        """Update the component info string."""
        parts = []
        if self.vad:
            parts.append(f"**VAD:** {self.vad.backend_name}")
        if self.stt:
            parts.append(f"**STT:** faster-whisper ({self.stt.model_size})")
        if self.llm:
            parts.append(f"**LLM:** {self.llm.model}")
        if self.tts:
            parts.append(f"**TTS:** kokoro-onnx ({self.tts.voice})")
        parts.append(f"**Memory:** {'on' if self.memory.memory_enabled else 'off'}")
        with self._ui_lock:
            self._last_component_info = " | ".join(parts)

    def get_ui_snapshot(self):
        """
        Read current shared UI state (called by Gradio timer).
        Returns a dict of all display values.
        """
        with self._ui_lock:
            return {
                "chat_history": list(self._chat_history),
                "transcription": self._last_transcription_md,
                "per_turn": self._last_per_turn_md,
                "session": self._last_session_md,
                "memory": self._last_memory_md,
                "component_info": self._last_component_info,
                "vad_confidence": self._last_vad_confidence,
                "tts_audio": self._pending_tts_audio,
                "update_counter": self._ui_update_counter,
            }

    def process_text_input(self, text: str) -> PipelineOutput:
        """
        Process a text input directly (skip mic/VAD/STT).
        Uses streaming LLM + sentence-chunked TTS, same as voice path.
        """
        self._processing = True
        pipeline_t0 = time.perf_counter()
        metrics = PipelineMetrics()
        metrics.stt_text = text
        metrics.stt_latency_ms = 0
        metrics.stt_confidence = 1.0
        metrics.stt_language = "en"
        metrics.speech_duration_ms = 0
        metrics.vad_backend = "text_input"

        output = PipelineOutput(metrics=metrics)

        try:
            self._set_status("thinking")

            tts_chunks: list[TTSResult] = []
            total_tts_ms = 0.0
            first_audio_time: Optional[float] = None

            def _on_sentence(sentence: str):
                nonlocal total_tts_ms, first_audio_time
                self._set_status("speaking")
                tts_result = self.tts.synthesize(sentence)
                tts_chunks.append(tts_result)
                total_tts_ms += tts_result.processing_time_ms
                if first_audio_time is None:
                    first_audio_time = time.perf_counter()

            llm_result = self.llm.generate_stream(
                self.memory, text, on_sentence=_on_sentence,
            )
            output.llm_result = llm_result

            metrics.llm_latency_ms = llm_result.processing_time_ms
            metrics.llm_response_text = llm_result.text
            metrics.llm_tokens = llm_result.token_count
            metrics.llm_prompt_tokens = llm_result.prompt_token_count
            metrics.llm_tokens_per_sec = llm_result.tokens_per_second
            metrics.llm_model = llm_result.model_name
            metrics.context_tokens_used = llm_result.context_tokens_used
            metrics.turns_in_context = llm_result.turns_in_context

            if tts_chunks:
                all_audio = np.concatenate([c.audio for c in tts_chunks])
                total_audio_dur = sum(c.audio_duration_seconds for c in tts_chunks)
                agg_tts = TTSResult(
                    audio=all_audio,
                    processing_time_ms=total_tts_ms,
                    audio_duration_seconds=total_audio_dur,
                    realtime_factor=total_audio_dur / (total_tts_ms / 1000) if total_tts_ms > 0 else 0,
                    voice_name=tts_chunks[0].voice_name,
                    sample_rate=tts_chunks[0].sample_rate,
                    text_length=len(llm_result.text),
                )
                output.tts_result = agg_tts

                metrics.tts_latency_ms = total_tts_ms
                metrics.tts_audio_duration_ms = total_audio_dur * 1000
                metrics.tts_realtime_factor = agg_tts.realtime_factor
                metrics.tts_voice = agg_tts.voice_name
                metrics.tts_sentences = len(tts_chunks)

            if first_audio_time is not None:
                metrics.time_to_first_audio_ms = (first_audio_time - pipeline_t0) * 1000

            metrics.compute_total()
            self.session_metrics.add_turn(metrics)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            output.error = str(e)

        self._processing = False
        return output

    # ---- Configuration ----

    def set_vad_threshold(self, threshold: float):
        self.vad_threshold = threshold
        if self.vad:
            self.vad.set_threshold(threshold)

    def set_silence_timeout(self, timeout_ms: float):
        self.silence_timeout_ms = timeout_ms

    def set_min_speech_duration(self, min_ms: float):
        self.min_speech_ms = min_ms

    def set_voice(self, voice: str):
        if self.tts:
            self.tts.set_voice(voice)

    def set_llm_model(self, model: str):
        if self.llm:
            self.llm.set_model(model)

    def set_system_prompt(self, prompt: str):
        self.memory.set_system_prompt(prompt)

    def set_max_context_tokens(self, max_tokens: int):
        self.memory.set_max_context_tokens(max_tokens)

    def set_memory_enabled(self, enabled: bool):
        """Toggle between stateful and stateless memory."""
        current_history = self.memory.get_full_history()
        current_prompt = self.memory.system_prompt
        max_tokens = self.memory.max_context_tokens

        if enabled:
            self.memory = ConversationMemory(
                system_prompt=current_prompt,
                max_context_tokens=max_tokens,
            )
        else:
            self.memory = StatelessMemory(
                system_prompt=current_prompt,
                max_context_tokens=max_tokens,
            )
        logger.info(f"Memory {'enabled' if enabled else 'disabled'}")

    def clear_conversation(self):
        """Reset conversation memory and shared UI chat history."""
        self.memory.clear()
        with self._ui_lock:
            self._chat_history.clear()
            self._last_transcription_md = "*Waiting for speech...*"
            self._last_per_turn_md = "*Metrics will appear after the first turn...*"
            self._last_memory_md = "**Memory:** No conversation yet"
            self._pending_tts_audio = None
            self._ui_update_counter += 1
        logger.info("Conversation cleared")

    def reset_audio_state(self):
        """Reset VAD/audio buffering state."""
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_frames = 0
        if self.vad:
            self.vad.reset()

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_processing(self) -> bool:
        return self._processing
