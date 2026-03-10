#!/usr/bin/env bash
set -e

echo "========================================="
echo "  AVA Voice Pipeline - Setup Script"
echo "========================================="
echo ""

# ---- Check Ollama ----
echo "[1/5] Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
else
    echo "  ✗ Ollama not found."
    echo "    Download from: https://ollama.com/download"
    echo "    Or: brew install ollama"
    exit 1
fi

# ---- Pull smollm2:135m ----
echo ""
echo "[2/5] Pulling smollm2:135m model..."
ollama pull smollm2:135m
echo "  ✓ smollm2:135m ready"

# ---- Check espeak-ng (required by kokoro-onnx for phonemization) ----
echo ""
echo "[3/5] Checking espeak-ng..."
if command -v espeak-ng &> /dev/null; then
    echo "  ✓ espeak-ng found: $(espeak-ng --version 2>/dev/null | head -1)"
else
    echo "  ✗ espeak-ng not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install espeak-ng
        echo "  ✓ espeak-ng installed"
    else
        echo "  ✗ Homebrew not found. Install espeak-ng manually:"
        echo "    brew install espeak-ng"
        exit 1
    fi
fi

# ---- Activate venv and install Python deps ----
echo ""
echo "[4/5] Installing Python dependencies..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "  ✓ Activated .venv"
else
    echo "  Creating .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "  ✓ Created and activated .venv"
fi

pip install --upgrade pip
pip install -r requirements.txt

# Install TEN VAD from git (native lib, separate from requirements.txt)
echo ""
echo "  Installing TEN VAD from git..."
pip install -U git+https://github.com/TEN-framework/ten-vad.git || {
    echo "  ⚠ TEN VAD install failed - will use Silero VAD fallback"
}

# ---- Verify installation ----
echo ""
echo "[5/5] Verifying installation..."

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
    checks.append(('ollama', '✓'))
except ImportError as e:
    checks.append(('ollama', f'✗ {e}'))

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

echo ""
echo "========================================="
echo "  Setup complete! Run with:"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo "========================================="
