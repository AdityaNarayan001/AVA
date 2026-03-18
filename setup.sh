#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  AVA — Voice AI Pipeline — First-Time Setup
#  Idempotent: safe to re-run at any time.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Colour

step() { echo -e "\n${BOLD}${CYAN}[$1]${NC} $2"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
err()  { echo -e "  ${RED}✗${NC} $1"; }

echo -e "${BOLD}"
echo "╔═══════════════════════════════════════════════╗"
echo "║     AVA — Voice AI Pipeline Setup             ║"
echo "╚═══════════════════════════════════════════════╝"
echo -e "${NC}"

# ─────────────────────────────────────────────────────────────
# Step 1: Choose LLM backend
# ─────────────────────────────────────────────────────────────
step "1/7" "Choose your LLM backend"
echo ""
echo "  AVA supports two LLM backends:"
echo ""
echo "    ${BOLD}1) Ollama${NC}  — run models locally (no internet needed at runtime)"
echo "       Best for: privacy, offline use, edge deployment"
echo "       Requires: Ollama installed + a pulled model"
echo ""
echo "    ${BOLD}2) LiteLLM${NC} — use any cloud/remote API (OpenAI, Anthropic, Juspay Grid,"
echo "       OpenRouter, Together, vLLM, etc.)"
echo "       Best for: powerful models, low-spec hardware"
echo "       Requires: an API key"
echo ""
echo -e "  ${YELLOW}You can change this anytime in config.yaml → llm.backend${NC}"
echo ""

while true; do
    read -rp "  Choose [1/2]: " llm_choice
    case "$llm_choice" in
        1) LLM_BACKEND="ollama"; break ;;
        2) LLM_BACKEND="openai_compatible"; break ;;
        *) echo "  Please enter 1 or 2." ;;
    esac
done

ok "LLM backend: $( [ "$LLM_BACKEND" = "ollama" ] && echo "Ollama (local)" || echo "LiteLLM (cloud API)" )"

# ─────────────────────────────────────────────────────────────
# Step 2: Ollama setup  OR  API key
# ─────────────────────────────────────────────────────────────
if [ "$LLM_BACKEND" = "ollama" ]; then
    step "2/7" "Setting up Ollama"

    if command -v ollama &> /dev/null; then
        ok "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
    else
        echo ""
        echo "  Ollama is not installed. Choose how to install:"
        echo "    ${BOLD}a)${NC} Homebrew  (brew install ollama)"
        echo "    ${BOLD}b)${NC} Official installer  (https://ollama.com/download)"
        echo ""

        while true; do
            read -rp "  Choose [a/b]: " install_choice
            case "$install_choice" in
                a|A)
                    if command -v brew &> /dev/null; then
                        echo "  Installing Ollama via Homebrew..."
                        brew install ollama
                        ok "Ollama installed"
                    else
                        err "Homebrew not found. Install Homebrew first: https://brew.sh"
                        exit 1
                    fi
                    break ;;
                b|B)
                    echo ""
                    echo "  1. Download and install Ollama from: https://ollama.com/download"
                    echo "  2. Re-run this script after installing."
                    exit 0 ;;
                *) echo "  Please enter a or b." ;;
            esac
        done
    fi

    # Ensure Ollama server is running
    if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  Starting Ollama server in background..."
        ollama serve &>/dev/null &
        sleep 2
        if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
            ok "Ollama server started"
        else
            warn "Could not start Ollama server. Start it manually: ollama serve"
        fi
    else
        ok "Ollama server is running"
    fi

    # Pull default model
    echo ""
    OLLAMA_MODEL="smollm2:135m"
    read -rp "  Model to pull (default: ${OLLAMA_MODEL}): " user_model
    [ -n "$user_model" ] && OLLAMA_MODEL="$user_model"

    echo "  Pulling ${OLLAMA_MODEL}..."
    ollama pull "$OLLAMA_MODEL"
    ok "${OLLAMA_MODEL} ready"

    # Create stub .env so ${AVA_API_KEY} placeholders don't cause warnings
    if [ ! -f .env ]; then
        cat > .env <<EOF
# AVA Environment Variables
# Add your API key here if you switch to LiteLLM later.
AVA_API_KEY=
EOF
        ok ".env stub created (add API key later if needed)"
    fi

else
    step "2/7" "Setting up LiteLLM (cloud API)"
    echo ""
    echo "  AVA needs an API key for the cloud LLM."
    echo "  This key is stored in ${BOLD}.env${NC} (git-ignored, never committed)."
    echo ""
    echo "  Supported providers:"
    echo "    • Juspay Grid  (https://grid.ai.juspay.net)"
    echo "    • OpenAI       (https://platform.openai.com)"
    echo "    • OpenRouter    (https://openrouter.ai)"
    echo "    • Together      (https://together.ai)"
    echo "    • Any OpenAI-compatible endpoint"
    echo ""

    read -rp "  Enter your API key: " api_key

    if [ -z "$api_key" ]; then
        warn "No API key entered. You can add it later to .env: AVA_API_KEY=your-key"
        api_key=""
    fi

    # Write .env
    cat > .env <<EOF
# AVA Environment Variables
# This file contains secrets — DO NOT commit to git.
AVA_API_KEY=${api_key}
EOF
    ok ".env created with API key"
    echo ""
    echo "  You can configure the base_url and model in config.yaml → llm.openai_compatible"
fi

# ─────────────────────────────────────────────────────────────
# Step 3: Update config.yaml with chosen backend
# ─────────────────────────────────────────────────────────────
step "3/7" "Configuring config.yaml"

if [ -f config.yaml ]; then
    # Update the backend field
    if command -v sed &> /dev/null; then
        sed -i '' "s/^  backend: .*/  backend: \"${LLM_BACKEND}\"/" config.yaml 2>/dev/null || true
        # If ollama, update only the ollama section's model line (first model: after "ollama:" header)
        if [ "$LLM_BACKEND" = "ollama" ] && [ -n "${OLLAMA_MODEL:-}" ]; then
            python3 -c "
import re, sys
with open('config.yaml') as f:
    text = f.read()
text = re.sub(
    r'(ollama:\n\s+model:\s+)\"[^\"]*\"',
    r'\\1\"${OLLAMA_MODEL}\"',
    text, count=1
)
with open('config.yaml', 'w') as f:
    f.write(text)
" 2>/dev/null || true
        fi
    fi
    ok "config.yaml updated (backend: ${LLM_BACKEND})"
else
    warn "config.yaml not found — defaults will be used"
fi

# ─────────────────────────────────────────────────────────────
# Step 4: espeak-ng (required by kokoro-onnx for phonemization)
# ─────────────────────────────────────────────────────────────
step "4/7" "Checking espeak-ng (TTS phonemizer)"

if command -v espeak-ng &> /dev/null; then
    ok "espeak-ng found: $(espeak-ng --version 2>/dev/null | head -1)"
else
    echo "  espeak-ng is required by kokoro-onnx for phonemization."
    if command -v brew &> /dev/null; then
        echo "  Installing via Homebrew..."
        brew install espeak-ng
        ok "espeak-ng installed"
    elif command -v apt-get &> /dev/null; then
        echo "  Installing via apt..."
        sudo apt-get update -qq && sudo apt-get install -y -qq espeak-ng
        ok "espeak-ng installed"
    else
        err "Could not install espeak-ng automatically."
        echo "    macOS: brew install espeak-ng"
        echo "    Linux: sudo apt install espeak-ng"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────
# Step 5: Python venv + dependencies
# ─────────────────────────────────────────────────────────────
step "5/7" "Setting up Python environment"

if [ -d ".venv" ]; then
    source .venv/bin/activate
    ok "Activated existing .venv"
else
    echo "  Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    ok "Created and activated .venv"
fi

echo "  Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# TEN VAD (native C lib, installed from git separately)
echo "  Installing TEN VAD from git..."
pip install -U git+https://github.com/TEN-framework/ten-vad.git -q 2>/dev/null || {
    warn "TEN VAD install failed — will use Silero VAD fallback"
}

ok "Python dependencies installed"

# ─────────────────────────────────────────────────────────────
# Step 6: Download TTS models (kokoro-onnx)
# ─────────────────────────────────────────────────────────────
step "6/7" "Downloading TTS models"

MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"

KOKORO_MODEL="$MODELS_DIR/kokoro-v1.0.onnx"
VOICES_FILE="$MODELS_DIR/voices-v1.0.bin"
HF_BASE="https://huggingface.co/hexgrad/Kokoro-82M-v1.0-ONNX/resolve/main"

download_model() {
    local url="$1" dest="$2" name="$3"
    if [ -f "$dest" ]; then
        ok "$name already exists ($(du -sh "$dest" | cut -f1 | xargs))"
    else
        echo "  Downloading $name..."
        curl -L --progress-bar -o "$dest" "$url"
        ok "$name downloaded ($(du -sh "$dest" | cut -f1 | xargs))"
    fi
}

download_model "${HF_BASE}/kokoro-v1.0.onnx"  "$KOKORO_MODEL" "kokoro-v1.0.onnx (~310 MB)"
download_model "${HF_BASE}/voices-v1.0.bin"    "$VOICES_FILE"  "voices-v1.0.bin (~27 MB)"

# ─────────────────────────────────────────────────────────────
# Step 7: Verify everything
# ─────────────────────────────────────────────────────────────
step "7/7" "Verifying installation"

python3 -c "
import sys

checks = []

try:
    import sounddevice
    checks.append(('sounddevice', '✓'))
except ImportError as e:
    checks.append(('sounddevice', f'✗ {e}'))

try:
    from faster_whisper import WhisperModel
    checks.append(('faster-whisper', '✓'))
except ImportError as e:
    checks.append(('faster-whisper', f'✗ {e}'))

try:
    import ollama
    checks.append(('ollama (client)', '✓'))
except ImportError as e:
    checks.append(('ollama (client)', f'✗ {e}'))

try:
    import kokoro_onnx
    checks.append(('kokoro-onnx', '✓'))
except ImportError as e:
    checks.append(('kokoro-onnx', f'✗ {e}'))

try:
    import gradio
    checks.append(('gradio', '✓'))
except ImportError as e:
    checks.append(('gradio', f'✗ {e}'))

try:
    from ten_vad import TenVad
    checks.append(('ten-vad', '✓'))
except Exception as e:
    checks.append(('ten-vad', f'⚠ fallback to silero ({e})'))

try:
    import torch
    checks.append(('torch (silero)', '✓'))
except ImportError as e:
    checks.append(('torch (silero)', f'✗ {e}'))

try:
    import yaml
    checks.append(('pyyaml', '✓'))
except ImportError as e:
    checks.append(('pyyaml', f'✗ {e}'))

try:
    import requests
    checks.append(('requests', '✓'))
except ImportError as e:
    checks.append(('requests', f'✗ {e}'))

import os
if os.path.exists('models/kokoro-v1.0.onnx') and os.path.exists('models/voices-v1.0.bin'):
    checks.append(('TTS models', '✓'))
else:
    checks.append(('TTS models', '✗ missing from models/'))

print()
for name, status in checks:
    print(f'  {status} {name}')

failures = [c for c in checks if c[1].startswith('✗')]
if failures:
    print(f'\n  {len(failures)} critical failure(s). Fix before running.')
    sys.exit(1)
else:
    print('\n  All dependencies ready!')
"

# ─────────────────────────────────────────────────────────────
# Done!
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║${NC}  ${GREEN}Setup complete!${NC}                              ${BOLD}║${NC}"
echo -e "${BOLD}╠═══════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}                                               ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Run AVA:                                     ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    source .venv/bin/activate                   ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    python terminal_app.py    ${CYAN}(Terminal UI)${NC}      ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}    python app.py             ${CYAN}(Gradio Web UI)${NC}    ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}                                               ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Config:  config.yaml                         ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Secrets: .env  (git-ignored)                  ${BOLD}║${NC}"
BACKEND_LABEL="$( [ "$LLM_BACKEND" = "ollama" ] && echo "Ollama (local)" || echo "LiteLLM (cloud API)" )"
echo -e "${BOLD}║${NC}  LLM:     ${CYAN}${BACKEND_LABEL}${NC}"
echo -e "${BOLD}║${NC}                                               ${BOLD}║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════╝${NC}"
