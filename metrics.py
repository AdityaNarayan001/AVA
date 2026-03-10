"""
Metrics collection and session tracking for AVA voice pipeline.
Tracks per-turn latency, token throughput, and cumulative session stats.
"""

import time
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics for a single conversation turn."""
    timestamp: float = 0.0
    turn_number: int = 0

    # VAD
    vad_confidence: float = 0.0
    vad_backend: str = ""
    speech_duration_ms: float = 0.0

    # STT
    stt_latency_ms: float = 0.0
    stt_text: str = ""
    stt_confidence: float = 0.0
    stt_language: str = ""
    stt_realtime_factor: float = 0.0

    # LLM
    llm_latency_ms: float = 0.0
    llm_response_text: str = ""
    llm_tokens: int = 0
    llm_prompt_tokens: int = 0
    llm_tokens_per_sec: float = 0.0
    llm_model: str = ""
    context_tokens_used: int = 0
    turns_in_context: int = 0

    # TTS
    tts_latency_ms: float = 0.0
    tts_audio_duration_ms: float = 0.0
    tts_realtime_factor: float = 0.0
    tts_voice: str = ""

    # Total
    total_latency_ms: float = 0.0  # end-to-end: speech end → TTS audio ready
    time_to_first_audio_ms: float = 0.0  # speech end → first TTS chunk playing (streaming)
    tts_sentences: int = 0  # number of TTS chunks in streaming mode

    def compute_total(self):
        """Calculate total E2E latency from component latencies."""
        self.total_latency_ms = self.stt_latency_ms + self.llm_latency_ms + self.tts_latency_ms

    def to_dict(self) -> Dict:
        return asdict(self)


def _safe_stats(values: List[float]) -> Dict[str, float]:
    """Compute stats for a list of values, handling empty lists."""
    if not values:
        return {"avg": 0, "min": 0, "max": 0, "p95": 0, "median": 0}
    sorted_v = sorted(values)
    p95_idx = int(len(sorted_v) * 0.95)
    p95_idx = min(p95_idx, len(sorted_v) - 1)
    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "p95": sorted_v[p95_idx],
        "median": statistics.median(values),
    }


class SessionMetrics:
    """
    Accumulates per-turn metrics across a session.
    Computes aggregate stats for evaluation.
    """

    def __init__(self):
        self.turns: List[PipelineMetrics] = []
        self.session_start: float = time.time()
        self._turn_counter: int = 0

    def add_turn(self, metrics: PipelineMetrics):
        """Record a completed turn's metrics."""
        self._turn_counter += 1
        metrics.turn_number = self._turn_counter
        metrics.timestamp = time.time()
        metrics.compute_total()
        self.turns.append(metrics)
        logger.info(
            f"Turn {self._turn_counter}: "
            f"STT={metrics.stt_latency_ms:.0f}ms "
            f"LLM={metrics.llm_latency_ms:.0f}ms "
            f"TTS={metrics.tts_latency_ms:.0f}ms "
            f"Total={metrics.total_latency_ms:.0f}ms"
        )

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def session_duration_seconds(self) -> float:
        return time.time() - self.session_start

    def get_summary(self) -> Dict:
        """Get aggregate session statistics."""
        if not self.turns:
            return {
                "turn_count": 0,
                "session_duration_s": self.session_duration_seconds,
                "stt_latency": _safe_stats([]),
                "llm_latency": _safe_stats([]),
                "tts_latency": _safe_stats([]),
                "total_latency": _safe_stats([]),
                "total_llm_tokens": 0,
                "avg_llm_tokens_per_sec": 0,
                "avg_context_tokens": 0,
            }

        stt_latencies = [t.stt_latency_ms for t in self.turns]
        llm_latencies = [t.llm_latency_ms for t in self.turns]
        tts_latencies = [t.tts_latency_ms for t in self.turns]
        total_latencies = [t.total_latency_ms for t in self.turns]

        total_tokens = sum(t.llm_tokens for t in self.turns)
        tok_speeds = [t.llm_tokens_per_sec for t in self.turns if t.llm_tokens_per_sec > 0]
        context_tokens = [t.context_tokens_used for t in self.turns if t.context_tokens_used > 0]

        return {
            "turn_count": len(self.turns),
            "session_duration_s": self.session_duration_seconds,
            "stt_latency": _safe_stats(stt_latencies),
            "llm_latency": _safe_stats(llm_latencies),
            "tts_latency": _safe_stats(tts_latencies),
            "total_latency": _safe_stats(total_latencies),
            "total_llm_tokens": total_tokens,
            "avg_llm_tokens_per_sec": statistics.mean(tok_speeds) if tok_speeds else 0,
            "avg_context_tokens": statistics.mean(context_tokens) if context_tokens else 0,
        }

    def get_last_turn(self) -> Optional[PipelineMetrics]:
        """Get the most recent turn's metrics."""
        return self.turns[-1] if self.turns else None

    def format_per_turn_metrics(self, metrics: PipelineMetrics) -> str:
        """Format a single turn's metrics as a readable string for the UI."""
        def color_latency(ms):
            if ms < 1000:
                return "🟢"
            elif ms < 3000:
                return "🟡"
            else:
                return "🔴"

        lines = [
            f"**Turn {metrics.turn_number}**",
            "",
            f"🎤 **Speech Duration:** {metrics.speech_duration_ms:.0f}ms",
            "",
            f"🗣️ **STT Latency:** {metrics.stt_latency_ms:.0f}ms | "
            f"Confidence: {metrics.stt_confidence:.2f} | "
            f"Language: {metrics.stt_language}",
            "",
            f"🧠 **LLM Latency:** {metrics.llm_latency_ms:.0f}ms | "
            f"Tokens: {metrics.llm_tokens} | "
            f"Speed: {metrics.llm_tokens_per_sec:.1f} tok/s",
            "",
            f"  Context: {metrics.context_tokens_used} tokens | "
            f"{metrics.turns_in_context} turns in context",
            "",
            f"🔊 **TTS Latency:** {metrics.tts_latency_ms:.0f}ms | "
            f"Audio: {metrics.tts_audio_duration_ms:.0f}ms | "
            f"RT Factor: {metrics.tts_realtime_factor:.1f}x"
            + (f" | {metrics.tts_sentences} chunks" if metrics.tts_sentences > 1 else ""),
            "",
            f"{color_latency(metrics.total_latency_ms)} **Total E2E Latency:** "
            f"{metrics.total_latency_ms:.0f}ms"
            + (f" | ⚡ **First Audio:** {metrics.time_to_first_audio_ms:.0f}ms"
               if metrics.time_to_first_audio_ms > 0 else ""),
        ]
        return "\n".join(lines)

    def format_session_summary(self) -> str:
        """Format cumulative session stats as markdown for the UI."""
        s = self.get_summary()
        if s["turn_count"] == 0:
            return "No turns completed yet."

        def fmt_stats(stats):
            return (
                f"Avg: {stats['avg']:.0f}ms | "
                f"Min: {stats['min']:.0f}ms | "
                f"Max: {stats['max']:.0f}ms | "
                f"P95: {stats['p95']:.0f}ms"
            )

        duration_min = s["session_duration_s"] / 60
        lines = [
            f"**Session: {s['turn_count']} turns | {duration_min:.1f} min**",
            "",
            "| Stage | Avg | Min | Max | P95 |",
            "|-------|-----|-----|-----|-----|",
            f"| STT | {s['stt_latency']['avg']:.0f}ms | {s['stt_latency']['min']:.0f}ms | {s['stt_latency']['max']:.0f}ms | {s['stt_latency']['p95']:.0f}ms |",
            f"| LLM | {s['llm_latency']['avg']:.0f}ms | {s['llm_latency']['min']:.0f}ms | {s['llm_latency']['max']:.0f}ms | {s['llm_latency']['p95']:.0f}ms |",
            f"| TTS | {s['tts_latency']['avg']:.0f}ms | {s['tts_latency']['min']:.0f}ms | {s['tts_latency']['max']:.0f}ms | {s['tts_latency']['p95']:.0f}ms |",
            f"| **Total** | **{s['total_latency']['avg']:.0f}ms** | **{s['total_latency']['min']:.0f}ms** | **{s['total_latency']['max']:.0f}ms** | **{s['total_latency']['p95']:.0f}ms** |",
            "",
            f"**LLM Throughput:** {s['total_llm_tokens']} total tokens | "
            f"{s['avg_llm_tokens_per_sec']:.1f} avg tok/s",
            "",
            f"**Avg Context:** {s['avg_context_tokens']:.0f} tokens",
        ]
        return "\n".join(lines)

    def format_latency_bar(self, metrics: PipelineMetrics) -> str:
        """Create an ASCII pipeline stage breakdown bar."""
        total = metrics.total_latency_ms
        if total <= 0:
            return ""
        stt_pct = metrics.stt_latency_ms / total * 100
        llm_pct = metrics.llm_latency_ms / total * 100
        tts_pct = metrics.tts_latency_ms / total * 100
        return (
            f"STT: {stt_pct:.0f}% | LLM: {llm_pct:.0f}% | TTS: {tts_pct:.0f}%"
        )

    def reset(self):
        """Clear all accumulated metrics."""
        self.turns.clear()
        self._turn_counter = 0
        self.session_start = time.time()
        logger.info("Session metrics reset")

    def export_json(self) -> str:
        """Export all per-turn metrics as JSON."""
        return json.dumps({
            "session_start": self.session_start,
            "session_duration_s": self.session_duration_seconds,
            "summary": self.get_summary(),
            "turns": [t.to_dict() for t in self.turns],
        }, indent=2, default=str)
