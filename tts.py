"""
TTS (Text-to-Speech) module for AVA.
Uses kokoro-onnx for lightweight, fast speech synthesis.
"""

import numpy as np
import time
import os
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "af_heart"
OUTPUT_SAMPLE_RATE = 24000
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(MODELS_DIR, "voices-v1.0.bin")


@dataclass
class TTSResult:
    """Complete TTS result with audio and metrics."""
    audio: np.ndarray  # float32 array at OUTPUT_SAMPLE_RATE
    processing_time_ms: float
    audio_duration_seconds: float
    realtime_factor: float  # audio_duration / processing_time (>1 = faster than RT)
    voice_name: str
    sample_rate: int
    text_length: int  # characters in input text


class TTSEngine:
    """
    Text-to-Speech engine using kokoro-onnx.
    Produces 24kHz audio from text input.
    """

    def __init__(self, voice: str = DEFAULT_VOICE):
        self.voice = voice
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self._model = None
        self._loaded = False
        logger.info(f"TTS engine configured: voice={voice}")

    def load(self):
        """Load the kokoro-onnx model. Call once at startup."""
        if self._loaded:
            return
        from kokoro_onnx import Kokoro
        logger.info("Loading kokoro-onnx model...")
        t0 = time.perf_counter()
        self._model = Kokoro(MODEL_PATH, VOICES_PATH)
        load_time = (time.perf_counter() - t0) * 1000
        self._loaded = True
        logger.info(f"TTS model loaded in {load_time:.0f}ms")

    def synthesize(self, text: str, voice: Optional[str] = None) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to speak
            voice: Optional voice override (default: self.voice)
            
        Returns:
            TTSResult with audio numpy array and metrics
        """
        if not self._loaded:
            self.load()

        use_voice = voice or self.voice
        text = text.strip()
        if not text:
            # Return silence for empty text
            silence = np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)
            return TTSResult(
                audio=silence,
                processing_time_ms=0,
                audio_duration_seconds=0.1,
                realtime_factor=0,
                voice_name=use_voice,
                sample_rate=self.sample_rate,
                text_length=0,
            )

        t0 = time.perf_counter()
        samples, sample_rate = self._model.create(
            text,
            voice=use_voice,
            speed=1.0,
            lang="en-us",
        )
        processing_time_ms = (time.perf_counter() - t0) * 1000

        # Ensure float32 numpy array
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float32)
        elif samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        audio_duration = len(samples) / sample_rate
        realtime_factor = audio_duration / (processing_time_ms / 1000) if processing_time_ms > 0 else 0

        result = TTSResult(
            audio=samples,
            processing_time_ms=processing_time_ms,
            audio_duration_seconds=audio_duration,
            realtime_factor=realtime_factor,
            voice_name=use_voice,
            sample_rate=sample_rate,
            text_length=len(text),
        )

        logger.info(
            f"TTS: {len(text)} chars → {audio_duration:.1f}s audio | "
            f"{processing_time_ms:.0f}ms | RT factor: {realtime_factor:.1f}x"
        )
        return result

    def set_voice(self, voice: str):
        """Change the active voice."""
        self.voice = voice
        logger.info(f"TTS voice changed to: {voice}")

    def get_available_voices(self):
        """Return list of available kokoro voice names."""
        return [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ]
