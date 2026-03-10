#!/usr/bin/env python3
"""
AVA — Cyberpunk Terminal Voice AI
Push-to-talk: Hold SPACE to record, release to process.
Fully local pipeline: VAD → STT → LLM → TTS → Speaker

Controls:
  SPACE (hold) — record speech
  Q            — quit
  C            — clear conversation
"""

import sys
import os
import tty
import termios
import select
import threading
import time
import platform
import logging
import numpy as np
import sounddevice as sd
import psutil
from scipy.signal import resample_poly
from math import gcd

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# ── Logging: suppress terminal output, log to file ──────────
logging.basicConfig(level=logging.WARNING, handlers=[logging.NullHandler()])
_fh = logging.FileHandler("ava.log", mode="w")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
logging.getLogger().addHandler(_fh)
logger = logging.getLogger("ava-tui")

from pipeline import VoicePipeline, SAMPLE_RATE, PipelineOutput, _safe_play

# ── Constants ────────────────────────────────────────────────
IS_APPLE_SILICON = platform.machine() == "arm64"
RELEASE_TIMEOUT = 0.15          # seconds after last space → "released"
RESOURCE_SCAN_INTERVAL = 5.0    # seconds between Ollama process scans
MAX_DISPLAY_MESSAGES = 12


class AVATerminal:
    """Cyberpunk push-to-talk terminal voice assistant."""

    def __init__(self):
        self.console = Console()
        self.pipeline = VoicePipeline()
        self.running = True

        # ── Recording ───────────────────────────
        self.recording = False
        self.processing = False
        self.audio_chunks: list = []
        self.audio_level: float = 0.0
        self.mic_stream = None
        self.record_start_time: float = 0

        # ── Keyboard ───────────────────────────
        self.space_held = False
        self.last_space_time: float = 0

        # ── Conversation ────────────────────────
        self.conversation: list = []        # [(role, text), ...]
        self.turn_count = 0
        self.session_start = time.time()

        # ── Metrics ────────────────────────────
        self.last_metrics = None            # PipelineMetrics
        self.last_output = None             # PipelineOutput

        # ── Status ─────────────────────────────
        self.status = "BOOT"
        self.status_detail = "Initializing..."

        # ── Resources ──────────────────────────
        self.process = psutil.Process()
        self.cpu_percent: float = 0
        self.ram_used_gb: float = 0
        self.ram_total_gb: float = 0
        self.ram_percent: float = 0
        self.process_ram_mb: float = 0
        self.ollama_ram_mb: float = 0
        self._last_ollama_scan: float = 0

    # ═══════════════════════════════════════════════════════════
    #  Pipeline status callback
    # ═══════════════════════════════════════════════════════════

    def _on_pipeline_status(self, status: str):
        mapping = {
            "transcribing": ("TRANSCRIBING", "Speech → text..."),
            "thinking":     ("THINKING",     "Generating response..."),
            "speaking":     ("SYNTHESIZING", "Text → speech..."),
            "loading_vad":  ("BOOT", "Loading VAD..."),
            "loading_stt":  ("BOOT", "Loading STT..."),
            "loading_llm":  ("BOOT", "Loading LLM..."),
            "loading_tts":  ("BOOT", "Loading TTS..."),
        }
        if status in mapping:
            self.status, self.status_detail = mapping[status]

    # ═══════════════════════════════════════════════════════════
    #  Resource Monitoring
    # ═══════════════════════════════════════════════════════════

    def _update_resources(self):
        try:
            self.cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self.ram_used_gb = mem.used / (1024 ** 3)
            self.ram_total_gb = mem.total / (1024 ** 3)
            self.ram_percent = mem.percent
            self.process_ram_mb = self.process.memory_info().rss / (1024 ** 2)

            now = time.time()
            if now - self._last_ollama_scan > RESOURCE_SCAN_INTERVAL:
                self._last_ollama_scan = now
                self.ollama_ram_mb = 0
                for proc in psutil.process_iter(["name", "memory_info"]):
                    try:
                        if "ollama" in (proc.info["name"] or "").lower():
                            self.ollama_ram_mb += proc.info["memory_info"].rss / (1024 ** 2)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════
    #  Recording (push-to-talk)
    # ═══════════════════════════════════════════════════════════

    def _start_recording(self):
        if self.recording or self.processing:
            return
        self.recording = True
        self.audio_chunks = []
        self.audio_level = 0
        self.record_start_time = time.time()
        self.status = "RECORDING"
        self.status_detail = "Listening..."
        self.mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype="int16", blocksize=1024,
            callback=self._audio_callback,
        )
        self.mic_stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            chunk = indata[:, 0].copy()
            self.audio_chunks.append(chunk)
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            self.audio_level = min(rms / 4000.0, 1.0)

    def _stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.audio_level = 0
        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None

        if not self.audio_chunks:
            self.status, self.status_detail = "READY", "No audio captured"
            return

        audio = np.concatenate(self.audio_chunks)
        dur_ms = len(audio) / SAMPLE_RATE * 1000
        if dur_ms < 300:
            self.status, self.status_detail = "READY", "Too short — hold longer"
            return

        self.processing = True
        self.status, self.status_detail = "PROCESSING", "Processing..."
        threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()

    def _process_audio(self, audio: np.ndarray):
        try:
            output = self.pipeline.process_utterance(audio)
            self.last_output = output
            self.last_metrics = output.metrics

            user_text = (output.metrics.stt_text or "").strip()
            assistant_text = (output.metrics.llm_response_text or "").strip()

            if user_text:
                self.conversation.append(("you", user_text))
                if assistant_text:
                    self.conversation.append(("ava", assistant_text))
                self.turn_count += 1

            # Play TTS
            if output.tts_result and output.tts_result.audio is not None:
                self.status, self.status_detail = "SPEAKING", "Playing response..."
                _safe_play(output.tts_result.audio, output.tts_result.sample_rate)

            self.status, self.status_detail = "READY", "Hold SPACE to speak"

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.status, self.status_detail = "ERROR", str(e)[:60]
        finally:
            self.processing = False

    # ═══════════════════════════════════════════════════════════
    #  Keyboard Input
    # ═══════════════════════════════════════════════════════════

    def _keyboard_loop(self):
        while self.running:
            try:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch == " ":
                        self.last_space_time = time.time()
                        if not self.space_held:
                            self.space_held = True
                            self._start_recording()
                    elif ch.lower() == "q":
                        self.running = False
                    elif ch.lower() == "c":
                        self.conversation.clear()
                        self.pipeline.clear_conversation()
                        self.pipeline.session_metrics.reset()
                        self.turn_count = 0
                        self.last_metrics = None
                        self.status_detail = "Cleared"
            except Exception:
                pass

    def _release_detector(self):
        while self.running:
            if self.space_held and time.time() - self.last_space_time > RELEASE_TIMEOUT:
                self.space_held = False
                self._stop_recording()
            time.sleep(0.04)

    # ═══════════════════════════════════════════════════════════
    #  Render Helpers
    # ═══════════════════════════════════════════════════════════

    def _bar(self, val: float, maxv: float, w: int = 10, style: str = "cyan") -> Text:
        ratio = min(val / maxv, 1.0) if maxv > 0 else 0
        filled = int(ratio * w)
        t = Text()
        t.append("█" * filled, style=style)
        t.append("░" * (w - filled), style="dim")
        return t

    def _color_for(self, val: float, good: float, warn: float) -> str:
        if val <= good:
            return "green"
        if val <= warn:
            return "yellow"
        return "red"

    # ── Status ──────────────────────────────────────────────

    def _build_status(self) -> Text:
        color_map = {
            "READY":        "green",
            "RECORDING":    "red",
            "PROCESSING":   "yellow",
            "TRANSCRIBING": "yellow",
            "THINKING":     "magenta",
            "SYNTHESIZING": "blue",
            "SPEAKING":     "cyan",
            "BOOT":         "yellow",
            "ERROR":        "red",
        }
        c = color_map.get(self.status, "white")
        t = Text()
        t.append("  ● ", style=f"bold {c}")
        t.append(self.status, style=f"bold {c}")
        t.append(f"  {self.status_detail}", style="dim")

        if self.recording:
            dur = time.time() - self.record_start_time
            t.append(f"    ⏺ {dur:.1f}s ", style="bold red")
            lev = self.audio_level
            bw = 15
            filled = int(lev * bw)
            t.append("▕", style="dim")
            t.append("█" * filled, style="red")
            t.append("░" * (bw - filled), style="dim red")
            t.append("▏", style="dim")
        return t

    # ── Conversation ────────────────────────────────────────

    def _build_conversation(self) -> Table:
        grid = Table.grid(padding=0, expand=True)
        grid.add_column()

        hdr = Text()
        hdr.append("  ── ", style="dim cyan")
        hdr.append("CONVERSATION", style="bold cyan")
        hdr.append(" ──────────────────────────────────────────────", style="dim cyan")
        grid.add_row(hdr)

        if not self.conversation:
            grid.add_row(Text("    Say something to begin...", style="dim italic"))
        else:
            for role, msg in self.conversation[-MAX_DISPLAY_MESSAGES:]:
                line = Text()
                if role == "you":
                    line.append("    YOU", style="bold cyan")
                    line.append(" › ", style="dim cyan")
                else:
                    line.append("    AVA", style="bold magenta")
                    line.append(" › ", style="dim magenta")
                line.append(msg)
                grid.add_row(line)
        return grid

    # ── Metrics ─────────────────────────────────────────────

    def _build_metrics(self) -> Table:
        grid = Table.grid(padding=0, expand=True)
        grid.add_column()

        hdr = Text()
        hdr.append("  ── ", style="dim cyan")
        hdr.append("LAST TURN", style="bold cyan")
        hdr.append(" ─────────────────────────────────────────────────", style="dim cyan")
        grid.add_row(hdr)

        if not self.last_metrics:
            grid.add_row(Text("    No turns yet", style="dim italic"))
            return grid

        m = self.last_metrics

        # Row 1: latencies
        t1 = Text("    ")
        t1.append("STT ", style="bold cyan")
        t1.append(f"{m.stt_latency_ms:.0f}ms")
        t1.append("  ▏  ", style="dim")
        t1.append("LLM ", style="bold magenta")
        t1.append(f"{m.llm_latency_ms:.0f}ms")
        t1.append("  ▏  ", style="dim")
        t1.append("TTS ", style="bold green")
        t1.append(f"{m.tts_latency_ms:.0f}ms")
        t1.append("  ▏  ", style="dim")
        t1.append("E2E ", style="bold")
        e2e_c = self._color_for(m.total_latency_ms, 2000, 4000)
        t1.append(f"{m.total_latency_ms:.0f}ms", style=f"bold {e2e_c}")
        grid.add_row(t1)

        # Row 2: detail metrics
        t2 = Text("    ")
        if m.stt_realtime_factor:
            t2.append(f"{m.stt_realtime_factor:.1f}x RT", style="dim cyan")
        t2.append("       ", style="dim")
        if m.llm_tokens_per_sec:
            t2.append(f"{m.llm_tokens_per_sec:.0f} tok/s", style="dim magenta")
        t2.append("      ", style="dim")
        if m.tts_realtime_factor:
            t2.append(f"{m.tts_realtime_factor:.1f}x RT", style="dim green")
        if m.llm_tokens:
            t2.append(f"        {m.llm_tokens} tokens", style="dim")
        grid.add_row(t2)

        # Row 3: pipeline breakdown bar
        total = m.total_latency_ms or 1
        stt_pct = m.stt_latency_ms / total
        llm_pct = m.llm_latency_ms / total
        tts_pct = m.tts_latency_ms / total
        bw = 50
        stt_w = max(1, int(stt_pct * bw))
        llm_w = max(1, int(llm_pct * bw))
        tts_w = max(1, bw - stt_w - llm_w)

        bar = Text("    ")
        bar.append("█" * stt_w, style="cyan")
        bar.append("█" * llm_w, style="magenta")
        bar.append("█" * tts_w, style="green")
        bar.append(f"  STT {stt_pct:.0%}", style="dim cyan")
        bar.append(f" │ LLM {llm_pct:.0%}", style="dim magenta")
        bar.append(f" │ TTS {tts_pct:.0%}", style="dim green")
        grid.add_row(bar)

        # Row 4: context info
        if m.context_tokens_used:
            ctx = Text("    ")
            ctx.append(f"Context: {m.context_tokens_used} tokens", style="dim")
            ctx.append(f"  ·  {m.turns_in_context} turns in memory", style="dim")
            ctx.append(f"  ·  Speech: {m.speech_duration_ms:.0f}ms", style="dim")
            grid.add_row(ctx)

        return grid

    # ── Session + System (combined for compactness) ─────────

    def _build_stats(self) -> Table:
        grid = Table.grid(padding=0, expand=True)
        grid.add_column()

        hdr = Text()
        hdr.append("  ── ", style="dim cyan")
        hdr.append("STATS", style="bold cyan")
        hdr.append(" ────────────────────────────────────────────────────", style="dim cyan")
        grid.add_row(hdr)

        # Session line
        uptime = time.time() - self.session_start
        mins, secs = divmod(int(uptime), 60)
        sess = Text("    ")
        sess.append(f"Turns: {self.turn_count}", style="bold")

        sm = self.pipeline.session_metrics
        e2e_vals = [t.total_latency_ms for t in sm.turns if t.total_latency_ms]
        if e2e_vals:
            avg = sum(e2e_vals) / len(e2e_vals)
            best = min(e2e_vals)
            worst = max(e2e_vals)
            sess.append(f"  ·  Avg: {avg:.0f}ms", style="dim")
            sess.append(f"  ·  Best: {best:.0f}ms", style="dim green")
            sess.append(f"  ·  Worst: {worst:.0f}ms", style="dim red")

        sess.append(f"  ·  Uptime: {mins}m {secs}s", style="dim")
        grid.add_row(sess)

        # System resources line
        sys_line = Text("    ")
        cpu_c = self._color_for(self.cpu_percent, 50, 80)
        sys_line.append("CPU ", style="dim")
        sys_line.append(f"{self.cpu_percent:4.0f}% ", style=f"bold {cpu_c}")
        sys_line.append_text(self._bar(self.cpu_percent, 100, 8, cpu_c))

        sys_line.append("  RAM ", style="dim")
        ram_c = self._color_for(self.ram_percent, 60, 85)
        sys_line.append(f"{self.ram_used_gb:.1f}/{self.ram_total_gb:.0f}GB ", style="bold")
        sys_line.append_text(self._bar(self.ram_percent, 100, 8, ram_c))
        sys_line.append(f" {self.ram_percent:.0f}%", style="dim")

        sys_line.append(f"  AVA: {self.process_ram_mb:.0f}MB", style="dim cyan")
        if self.ollama_ram_mb:
            sys_line.append(f"  Ollama: {self.ollama_ram_mb:.0f}MB", style="dim magenta")
        grid.add_row(sys_line)

        # GPU / VRAM line
        gpu = Text("    ")
        if IS_APPLE_SILICON:
            gpu.append("GPU/VRAM ", style="dim")
            gpu.append("Apple Silicon Unified Memory ", style="dim italic")
            gpu.append("(shared with RAM above)", style="dim")
        else:
            gpu.append("GPU ", style="dim")
            gpu.append("Integrated — no dedicated VRAM", style="dim italic")
        grid.add_row(gpu)

        # Pipeline config line
        pipe = Text("    ")
        if self.pipeline.vad:
            pipe.append("VAD:", style="dim")
            pipe.append(f"{self.pipeline.vad.backend_name} ", style="cyan")
        if self.pipeline.stt:
            pipe.append("STT:", style="dim")
            pipe.append(f"whisper-{self.pipeline.stt.model_size} ", style="cyan")
        if self.pipeline.llm:
            pipe.append("LLM:", style="dim")
            pipe.append(f"{self.pipeline.llm.model} ", style="magenta")
        if self.pipeline.tts:
            pipe.append("TTS:", style="dim")
            pipe.append(f"kokoro({self.pipeline.tts.voice}) ", style="green")
        grid.add_row(pipe)

        return grid

    # ── Footer ──────────────────────────────────────────────

    def _build_footer(self) -> Text:
        t = Text("  ")
        t.append(" SPACE ", style="bold white on dark_green")
        t.append(" record  ", style="dim")
        t.append(" Q ", style="bold white on red")
        t.append(" quit  ", style="dim")
        t.append(" C ", style="bold white on blue")
        t.append(" clear ", style="dim")
        return t

    # ═══════════════════════════════════════════════════════════
    #  Main Render
    # ═══════════════════════════════════════════════════════════

    def _render(self) -> Panel:
        grid = Table.grid(padding=0, expand=True)
        grid.add_column()

        grid.add_row(self._build_status())
        grid.add_row(Text(""))
        grid.add_row(self._build_conversation())
        grid.add_row(Text(""))
        grid.add_row(self._build_metrics())
        grid.add_row(Text(""))
        grid.add_row(self._build_stats())
        grid.add_row(Text(""))
        grid.add_row(self._build_footer())

        return Panel(
            grid,
            title="[bold cyan]░▒▓[/] [bold white]A V A[/] [bold cyan]▓▒░[/]  [dim]//[/]  [bold magenta]VOICE AI PIPELINE[/]",
            subtitle="[dim cyan]LOCAL  ·  REAL-TIME  ·  EDGE[/]",
            border_style="bright_cyan",
            box=box.DOUBLE,
            expand=True,
        )

    # ═══════════════════════════════════════════════════════════
    #  Main Loop
    # ═══════════════════════════════════════════════════════════

    def run(self):
        if not sys.stdin.isatty():
            print("Error: interactive terminal required.")
            sys.exit(1)

        self.console.clear()
        self.console.print(
            Panel(
                "[bold cyan]░▒▓ A V A ▓▒░[/]\n\n"
                "[dim]Initializing voice pipeline — this takes a few seconds...[/]",
                border_style="cyan", box=box.DOUBLE,
            )
        )

        # Load models
        self.pipeline._on_status_change = self._on_pipeline_status
        try:
            self.pipeline.initialize()
        except Exception as e:
            self.console.print(f"\n[bold red]Failed to initialize pipeline:[/] {e}")
            sys.exit(1)

        self.status, self.status_detail = "READY", "Hold SPACE to speak"
        self._update_resources()

        # Save terminal settings and switch to cbreak mode
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            # Start input threads
            threading.Thread(target=self._keyboard_loop, daemon=True).start()
            threading.Thread(target=self._release_detector, daemon=True).start()

            # Render loop
            self.console.clear()
            with Live(
                self._render(),
                refresh_per_second=4,
                console=self.console,
                screen=False,
            ) as live:
                while self.running:
                    self._update_resources()
                    live.update(self._render())
                    time.sleep(0.25)

        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup
            if self.mic_stream:
                try:
                    self.mic_stream.stop()
                    self.mic_stream.close()
                except Exception:
                    pass
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.console.clear()
            self.console.print("[dim]AVA terminated. Logs in ava.log[/]")


# ═════════════════════════════════════════════════════════════
#  Entry Point
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = AVATerminal()
    app.run()
