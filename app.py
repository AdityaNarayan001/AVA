"""
AVA - Voice AI Pipeline Test App (Real-Time Mode)
Gradio-based UI for testing a local voice AI stack:
  Mic → TEN VAD → faster-whisper (tiny) → smollm2:135m → kokoro-onnx → Speaker

The mic listens continuously. VAD auto-detects when you speak.
When you stop speaking, AVA transcribes → thinks → replies aloud.
No buttons needed — just talk naturally.
"""

import gradio as gr
import numpy as np
import threading
import time
import json
import logging
import sys
import os
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ava")

from pipeline import VoicePipeline, SAMPLE_RATE, PipelineOutput
from memory import DEFAULT_SYSTEM_PROMPT

# ─────────────────────────────────────────────────────────────
# Global pipeline instance
# ─────────────────────────────────────────────────────────────
pipeline = VoicePipeline()

# Track UI state version so timer only pushes real changes
_last_seen_counter = -1


# ─────────────────────────────────────────────────────────────
# Status mapping for display
# ─────────────────────────────────────────────────────────────
STATUS_DISPLAY = {
    "idle": "⚪ Idle — click **Start Listening** to begin",
    "listening": "🟢 Listening — speak naturally...",
    "speech_detected": "🎤 Speech detected — keep talking...",
    "transcribing": "🗣️ Transcribing your speech...",
    "thinking": "🧠 Thinking...",
    "speaking": "🔊 AVA is speaking...",
    "loading_vad": "⏳ Loading VAD model...",
    "loading_stt": "⏳ Loading STT model...",
    "loading_llm": "⏳ Loading LLM...",
    "loading_tts": "⏳ Loading TTS model...",
    "error": "🔴 Error — check terminal logs",
}


# ─────────────────────────────────────────────────────────────
# Start / Stop listening
# ─────────────────────────────────────────────────────────────

def toggle_listening(is_listening):
    """Toggle continuous mic listening on/off."""
    if is_listening:
        # Currently listening → stop
        pipeline.stop_listening()
        return (
            gr.update(value="▶️ Start Listening", variant="primary"),
            STATUS_DISPLAY["idle"],
            False,
        )
    else:
        # Not listening → start
        pipeline.start_listening()
        return (
            gr.update(value="⏹️ Stop Listening", variant="stop"),
            STATUS_DISPLAY["listening"],
            True,
        )


# ─────────────────────────────────────────────────────────────
# Timer-driven UI refresh
# ─────────────────────────────────────────────────────────────

def poll_ui_updates():
    """
    Called every 500ms by gr.Timer.
    Reads the shared pipeline state and returns updated UI values.
    """
    global _last_seen_counter

    snap = pipeline.get_ui_snapshot()
    status = STATUS_DISPLAY.get(pipeline.status, pipeline.status)

    # VAD confidence bar
    vad_conf = snap["vad_confidence"]
    bar_len = int(vad_conf * 20)
    vad_bar = f"VAD: {'█' * bar_len}{'░' * (20 - bar_len)} {vad_conf:.2f}"

    counter = snap["update_counter"]
    if counter == _last_seen_counter:
        # No new data — still update status + VAD for liveliness
        return (
            gr.update(),   # chat — no change
            gr.update(),   # transcription
            gr.update(),   # per_turn
            gr.update(),   # session
            status,        # status (always update)
            gr.update(),   # memory
            gr.update(),   # component info
            vad_bar,       # vad (always update)
        )

    _last_seen_counter = counter

    return (
        snap["chat_history"],
        snap["transcription"],
        snap["per_turn"],
        snap["session"],
        status,
        snap["memory"],
        snap["component_info"],
        vad_bar,
    )


# ─────────────────────────────────────────────────────────────
# Text input (bypass mic, still uses LLM + TTS)
# ─────────────────────────────────────────────────────────────

def process_text_input(text):
    """Process typed text: LLM → TTS → speaker. Updates shared state."""
    if not text or not text.strip():
        return ""

    # Ensure models are loaded even if mic isn't running
    if pipeline.vad is None:
        pipeline.initialize()
        pipeline._update_component_info()

    output = pipeline.process_text_input(text.strip())
    pipeline._update_ui_state(output)

    # Play TTS through speakers
    if output.tts_result and output.tts_result.audio is not None:
        import sounddevice as sd
        try:
            sd.play(output.tts_result.audio, samplerate=output.tts_result.sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Playback error: {e}")

    return ""  # Clear the text input


# ─────────────────────────────────────────────────────────────
# Settings callbacks
# ─────────────────────────────────────────────────────────────

def update_vad_threshold(value):
    pipeline.set_vad_threshold(value)
    return f"VAD threshold: {value:.2f}"

def update_silence_timeout(value):
    pipeline.set_silence_timeout(value * 1000)
    return f"Silence timeout: {value:.1f}s"

def update_min_speech(value):
    pipeline.set_min_speech_duration(value * 1000)
    return f"Min speech: {value:.1f}s"

def update_voice(voice):
    pipeline.set_voice(voice)
    return f"Voice: {voice}"

def update_llm_model(model):
    pipeline.set_llm_model(model)
    return f"Model: {model}"

def update_system_prompt(prompt):
    pipeline.set_system_prompt(prompt)
    return f"System prompt updated ({len(prompt)} chars)"

def update_max_context_tokens(value):
    pipeline.set_max_context_tokens(int(value))
    return f"Max context tokens: {int(value)}"

def toggle_memory(enabled):
    pipeline.set_memory_enabled(enabled)
    return f"Memory {'enabled' if enabled else 'disabled'}"


def clear_conversation():
    pipeline.clear_conversation()
    pipeline.session_metrics.reset()
    return (
        [],
        "*Waiting for speech...*",
        "*Metrics will appear after the first turn...*",
        "No turns completed yet.",
        "**Memory:** No conversation yet",
    )


def reset_metrics():
    pipeline.session_metrics.reset()
    return ("*Metrics will appear after the first turn...*", "No turns completed yet.")


def export_metrics():
    """Export metrics as a downloadable JSON file."""
    data = pipeline.session_metrics.export_json()
    full_export = json.loads(data)
    full_export["conversation"] = json.loads(pipeline.memory.export_json())

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_ava_metrics.json", delete=False, prefix="ava_"
    )
    json.dump(full_export, tmp, indent=2)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
.metrics-card {
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}
.status-bar {
    font-size: 1.2em;
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 8px;
    text-align: center;
}
.component-info {
    font-size: 0.85em;
    color: #666;
    padding: 4px 8px;
}
.vad-bar {
    font-family: monospace;
    font-size: 1.1em;
    padding: 4px 8px;
}
"""


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="AVA - Voice AI Pipeline Tester") as app:

        # Hidden state to track if we're listening
        listening_state = gr.State(False)

        # ── Header ──────────────────────────────────────────
        gr.Markdown(
            "# 🎙️ AVA — Real-Time Voice AI Pipeline Tester\n"
            "**Mic → TEN VAD → faster-whisper (tiny) → smollm2:135m → kokoro-onnx → Speaker**\n\n"
            "Click **Start Listening**, then just talk. AVA detects when you speak, "
            "transcribes, thinks, and replies through your speakers — hands-free."
        )

        # Status bar
        status_display = gr.Markdown(
            value=STATUS_DISPLAY["idle"],
            elem_classes=["status-bar"],
        )

        # Component info bar
        component_info = gr.Markdown(
            value="*Models will load when listening starts...*",
            elem_classes=["component-info"],
        )

        # ── Main control ────────────────────────────────────
        with gr.Row():
            listen_btn = gr.Button(
                "▶️ Start Listening",
                variant="primary",
                scale=2,
            )
            vad_display = gr.Textbox(
                value="VAD: ░░░░░░░░░░░░░░░░░░░░ 0.00",
                label="Voice Activity",
                interactive=False,
                scale=3,
                elem_classes=["vad-bar"],
            )

        # ── Main Layout ─────────────────────────────────────
        with gr.Row():
            # ============ LEFT COLUMN: Conversation ============
            with gr.Column(scale=3):

                gr.Markdown("### 💬 Conversation")
                chatbot = gr.Chatbot(
                    value=[],
                    height=420,
                    placeholder="Start listening and speak — conversation appears here automatically...",
                )

                # Text input as fallback
                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="Or type a message and press Enter...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="secondary", scale=1)

                # Memory info bar
                memory_info = gr.Markdown(value="**Memory:** No conversation yet")

                # Conversation controls
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Conversation", size="sm")

            # ============ RIGHT COLUMN: Metrics ============
            with gr.Column(scale=2):

                # Section B: Live Transcription
                gr.Markdown("### 🗣️ Transcription")
                transcription_display = gr.Markdown(
                    value="*Waiting for speech...*",
                    elem_classes=["metrics-card"],
                )

                # Section C: Per-Turn Metrics
                gr.Markdown("### ⏱️ Per-Turn Metrics")
                per_turn_display = gr.Markdown(
                    value="*Metrics will appear after the first turn...*",
                    elem_classes=["metrics-card"],
                )

                # Section D: Session Summary
                gr.Markdown("### 📊 Session Summary")
                session_display = gr.Markdown(
                    value="No turns completed yet.",
                    elem_classes=["metrics-card"],
                )

                # Metrics controls
                with gr.Row():
                    reset_metrics_btn = gr.Button("🔄 Reset Metrics", size="sm")
                    export_btn = gr.Button("📥 Export Metrics", size="sm")

                export_file = gr.File(label="Exported Metrics", visible=False)

        # ── Section E: Settings ─────────────────────────────
        with gr.Accordion("⚙️ Settings", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**VAD Settings**")
                    vad_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                        label="VAD Threshold",
                        info="Higher = more aggressive speech detection",
                    )
                    silence_timeout = gr.Slider(
                        minimum=0.3, maximum=3.0, value=1.0, step=0.1,
                        label="Silence Timeout (seconds)",
                        info="How long to wait after speech stops",
                    )
                    min_speech = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                        label="Min Speech Duration (seconds)",
                        info="Ignore utterances shorter than this",
                    )

                with gr.Column():
                    gr.Markdown("**Voice & Model**")
                    voice_select = gr.Dropdown(
                        choices=[
                            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
                            "am_adam", "am_michael",
                            "bf_emma", "bf_isabella",
                            "bm_george", "bm_lewis",
                        ],
                        value="af_heart",
                        label="TTS Voice",
                    )
                    llm_model_select = gr.Dropdown(
                        choices=["smollm2:135m", "smollm2:360m", "smollm2:1.7b"],
                        value="smollm2:135m",
                        label="LLM Model",
                    )

                with gr.Column():
                    gr.Markdown("**Memory Settings**")
                    memory_toggle = gr.Checkbox(
                        value=True,
                        label="Enable Conversation Memory",
                        info="When off, each turn is stateless",
                    )
                    max_context_tokens = gr.Slider(
                        minimum=500, maximum=2000, value=1500, step=50,
                        label="Max Context Tokens",
                        info="Token budget for conversation history",
                    )
                    system_prompt = gr.Textbox(
                        value=DEFAULT_SYSTEM_PROMPT,
                        label="System Prompt",
                        lines=3,
                        max_lines=6,
                    )

            settings_status = gr.Markdown(value="")

        # ── Timer for live UI updates ───────────────────────
        timer = gr.Timer(value=0.5, active=True)

        timer.tick(
            fn=poll_ui_updates,
            outputs=[
                chatbot,
                transcription_display,
                per_turn_display,
                session_display,
                status_display,
                memory_info,
                component_info,
                vad_display,
            ],
        )

        # ── Event Handlers ──────────────────────────────────

        # Start / Stop listening
        listen_btn.click(
            fn=toggle_listening,
            inputs=[listening_state],
            outputs=[listen_btn, status_display, listening_state],
        )

        # Text input → LLM + TTS (plays through speaker)
        send_btn.click(
            fn=process_text_input,
            inputs=[text_input],
            outputs=[text_input],
        )
        text_input.submit(
            fn=process_text_input,
            inputs=[text_input],
            outputs=[text_input],
        )

        # Clear conversation
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, transcription_display, per_turn_display, session_display, memory_info],
        )

        # Reset metrics
        reset_metrics_btn.click(
            fn=reset_metrics,
            outputs=[per_turn_display, session_display],
        )

        # Export metrics
        export_btn.click(
            fn=export_metrics,
            outputs=[export_file],
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[export_file],
        )

        # Settings
        vad_threshold.change(fn=update_vad_threshold, inputs=[vad_threshold], outputs=[settings_status])
        silence_timeout.change(fn=update_silence_timeout, inputs=[silence_timeout], outputs=[settings_status])
        min_speech.change(fn=update_min_speech, inputs=[min_speech], outputs=[settings_status])
        voice_select.change(fn=update_voice, inputs=[voice_select], outputs=[settings_status])
        llm_model_select.change(fn=update_llm_model, inputs=[llm_model_select], outputs=[settings_status])
        system_prompt.change(fn=update_system_prompt, inputs=[system_prompt], outputs=[settings_status])
        max_context_tokens.change(fn=update_max_context_tokens, inputs=[max_context_tokens], outputs=[settings_status])
        memory_toggle.change(fn=toggle_memory, inputs=[memory_toggle], outputs=[settings_status])

    return app


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting AVA Real-Time Voice Pipeline Tester...")
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
