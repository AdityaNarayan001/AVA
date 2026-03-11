"""
STT (Speech-to-Text) module for AVA.
Uses faster-whisper with the tiny model for minimal latency.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional
import logging

from config import get_config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class STTSegment:
    """A single transcription segment with timing and confidence."""
    text: str
    start: float  # seconds
    end: float  # seconds
    avg_logprob: float
    no_speech_prob: float


@dataclass
class STTResult:
    """Complete STT result with metrics."""
    text: str
    language: str
    language_probability: float
    duration_seconds: float  # input audio duration
    processing_time_ms: float
    segments: List[STTSegment] = field(default_factory=list)

    @property
    def realtime_factor(self) -> float:
        """How much faster than real-time. >1 means faster."""
        if self.processing_time_ms <= 0:
            return 0.0
        return (self.duration_seconds * 1000) / self.processing_time_ms


class STTEngine:
    """
    Speech-to-Text engine using faster-whisper (CTranslate2 backend).
    Optimized for low-latency CPU inference on macOS.
    """

    def __init__(self, model_size: str = None, device: str = None, compute_type: str = None):
        cfg = get_config().stt
        self.model_size = model_size or cfg.model_size
        self.device = device or cfg.device
        self.compute_type = compute_type or cfg.compute_type
        self._language = cfg.language
        self._beam_size = cfg.beam_size
        self._vad_filter = cfg.vad_filter
        self.model = None
        self._loaded = False
        logger.info(f"STT engine configured: model={self.model_size}, device={self.device}, compute_type={self.compute_type}")

    def load(self):
        """Load the whisper model. Call once at startup."""
        if self._loaded:
            return
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper model '{self.model_size}'...")
        t0 = time.perf_counter()
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        load_time = (time.perf_counter() - t0) * 1000
        self._loaded = True
        logger.info(f"STT model loaded in {load_time:.0f}ms")

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> STTResult:
        """
        Transcribe audio numpy array to text.
        
        Args:
            audio: int16 or float32 numpy array at the given sample rate
            sample_rate: Audio sample rate (default 16000)
            
        Returns:
            STTResult with transcription, timing, and confidence metrics
        """
        if not self._loaded:
            self.load()

        # Compute audio duration
        duration_seconds = len(audio) / sample_rate

        # Convert int16 to float32 if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_float = audio
        else:
            audio_float = audio.astype(np.float32)

        # Sanitize: replace NaN/Inf, skip near-silent audio
        audio_float = np.nan_to_num(audio_float, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_float = np.clip(audio_float, -1.0, 1.0)

        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms < 1e-6 or np.all(audio_float == 0):
            logger.info("STT: audio is silent, skipping transcription")
            return STTResult(
                text="",
                language="en",
                language_probability=1.0,
                duration_seconds=duration_seconds,
                processing_time_ms=0.0,
                segments=[],
            )

        # Pass numpy array directly to faster-whisper (no temp file I/O)
        t0 = time.perf_counter()
        lang = self._language if self._language != "auto" else None
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            segments_iter, info = self.model.transcribe(
                audio_float,
                beam_size=self._beam_size,
                best_of=1,
                language=lang,
                vad_filter=self._vad_filter,
                without_timestamps=False,
            )

            # Collect segments (iterator is lazy, so keep inside errstate)
            segments = []
            full_text_parts = []
            for seg in segments_iter:
                segments.append(STTSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                ))
                full_text_parts.append(seg.text.strip())

        processing_time_ms = (time.perf_counter() - t0) * 1000

        full_text = " ".join(full_text_parts).strip()

        result = STTResult(
            text=full_text,
            language=info.language,
            language_probability=info.language_probability,
            duration_seconds=duration_seconds,
            processing_time_ms=processing_time_ms,
            segments=segments,
        )

        logger.info(
            f"STT: '{full_text[:60]}...' | "
            f"{processing_time_ms:.0f}ms | "
            f"lang={info.language} ({info.language_probability:.2f})"
        )
        return result
