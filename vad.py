"""
VAD (Voice Activity Detection) module for AVA.
Primary: TEN VAD | Fallback: Silero VAD
Provides a unified interface for speech detection with confidence scores.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class VADResult:
    """Result from a single VAD frame processing."""
    is_speech: bool
    confidence: float  # 0.0 - 1.0
    backend: str  # "ten_vad" or "silero_vad"


class TenVADBackend:
    """TEN VAD backend using native C library."""

    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        from ten_vad import TenVad
        self.vad = TenVad(hop_size=hop_size, threshold=threshold)
        self.hop_size = hop_size
        self.threshold = threshold
        self.name = "ten_vad"
        logger.info(f"TEN VAD initialized (hop_size={hop_size}, threshold={threshold})")

    def process(self, audio_chunk: np.ndarray) -> VADResult:
        """Process a single audio frame. Expects int16 array of length hop_size."""
        result = self.vad.process(audio_chunk)
        # ten_vad.process() returns a tuple: (probability: float, flag: int)
        probability, flag = result
        return VADResult(
            is_speech=bool(flag),
            confidence=float(probability),
            backend=self.name,
        )

    def set_threshold(self, threshold: float):
        self.threshold = threshold
        # TEN VAD threshold is set at init; recreate if needed
        from ten_vad import TenVad
        self.vad = TenVad(hop_size=self.hop_size, threshold=threshold)


class SileroVADBackend:
    """Silero VAD backend using PyTorch."""

    def __init__(self, threshold: float = 0.5):
        import torch
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        self.threshold = threshold
        self.name = "silero_vad"
        self._sample_rate = 16000
        logger.info(f"Silero VAD initialized (threshold={threshold})")

    def process(self, audio_chunk: np.ndarray) -> VADResult:
        """Process audio chunk. Expects int16 numpy array."""
        import torch
        # Silero expects float32 tensor normalized to [-1, 1]
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float)
        confidence = self.model(tensor, self._sample_rate).item()
        return VADResult(
            is_speech=confidence >= self.threshold,
            confidence=confidence,
            backend=self.name,
        )

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def reset(self):
        """Reset Silero VAD internal state (call between utterances)."""
        self.model.reset_states()


class VADProcessor:
    """
    Unified VAD processor with automatic fallback.
    Tries TEN VAD first, falls back to Silero VAD if unavailable.
    """

    def __init__(self, hop_size: int = 512, threshold: float = 0.5):
        self.hop_size = hop_size
        self.threshold = threshold
        self.backend: Optional[object] = None
        self._init_backend()

    def _init_backend(self):
        """Try TEN VAD, fallback to Silero VAD."""
        # Try TEN VAD first
        try:
            self.backend = TenVADBackend(
                hop_size=self.hop_size,
                threshold=self.threshold,
            )
            logger.info("Using TEN VAD backend")
            return
        except Exception as e:
            logger.warning(f"TEN VAD unavailable ({e}), trying Silero VAD...")

        # Fallback to Silero VAD
        try:
            self.backend = SileroVADBackend(threshold=self.threshold)
            logger.info("Using Silero VAD backend (fallback)")
            return
        except Exception as e:
            logger.error(f"Silero VAD also failed: {e}")
            raise RuntimeError("No VAD backend available. Install ten-vad or silero-vad+torch.")

    def process(self, audio_chunk: np.ndarray) -> VADResult:
        """
        Process an audio chunk through the active VAD backend.
        
        Args:
            audio_chunk: int16 numpy array at 16kHz
            
        Returns:
            VADResult with is_speech, confidence, and backend name
        """
        return self.backend.process(audio_chunk)

    def set_threshold(self, threshold: float):
        """Update VAD threshold."""
        self.threshold = threshold
        if self.backend:
            self.backend.set_threshold(threshold)

    def reset(self):
        """Reset VAD state (call between utterances)."""
        if hasattr(self.backend, 'reset'):
            self.backend.reset()

    @property
    def backend_name(self) -> str:
        return self.backend.name if self.backend else "none"
