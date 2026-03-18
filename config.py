"""
AVA Configuration Loader.
Reads config.yaml and provides typed access to all pipeline parameters.
Falls back to sensible defaults if config.yaml is missing or incomplete.
Secrets are loaded from .env and injected via ${VAR_NAME} placeholders.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE_DIR, "config.yaml")
_ENV_PATH = os.path.join(_BASE_DIR, ".env")

# ── .env loader ──────────────────────────────────────────────

def _load_dotenv(path: str = _ENV_PATH):
    """Load .env file into os.environ (lightweight, no dependency)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    logger.debug(f"Loaded .env from {path}")


_ENV_VAR_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def _resolve_env_vars(obj):
    """Recursively replace ${VAR_NAME} placeholders with env values."""
    if isinstance(obj, str):
        def _replace(m):
            var = m.group(1)
            val = os.environ.get(var)
            if val is None:
                logger.warning(f"Env var ${{{var}}} not set — keeping placeholder")
                return m.group(0)
            return val
        return _ENV_VAR_RE.sub(_replace, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


# ── Dataclasses mirroring the YAML structure ─────────────────


@dataclass
class OllamaConfig:
    model: str = "smollm2:135m"
    host: str = "http://localhost:11434"
    keep_alive: str = "10m"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_ctx: int = 2048
    repeat_penalty: float = 1.1


@dataclass
class OpenAICompatibleConfig:
    base_url: str = "https://grid.ai.juspay.net/v1"
    api_key: str = "API-KEY"
    model: str = "kimi-latest"
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    extra_headers: Dict[str, str] = field(default_factory=lambda: {"x-api-key": "API-KEY"})
    timeout: int = 30


@dataclass
class LLMConfig:
    backend: str = "ollama"  # "ollama" | "openai_compatible"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai_compatible: OpenAICompatibleConfig = field(default_factory=OpenAICompatibleConfig)


@dataclass
class STTConfig:
    model_size: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1
    vad_filter: bool = False


@dataclass
class TTSConfig:
    voice: str = "af_heart"
    speed: float = 1.0
    language: str = "en-us"
    model_path: str = "models/kokoro-v1.0.onnx"
    voices_path: str = "models/voices-v1.0.bin"


@dataclass
class VADConfig:
    threshold: float = 0.5
    silence_timeout_ms: int = 1000
    min_speech_ms: int = 500
    frame_ms: int = 32
    backend: str = "auto"


@dataclass
class MemoryConfig:
    enabled: bool = True
    max_context_tokens: int = 1500
    tokens_per_word: float = 1.3
    system_prompt: str = (
        "You are AVA, a helpful and concise voice assistant running on a small device. "
        "Keep your responses short, natural, and conversational — typically 1-3 sentences. "
        "Speak in plain language. No markdown, no bullet points, no numbered lists. "
        "If you don't know something, say so briefly."
    )


@dataclass
class StreamingConfig:
    enabled: bool = True
    min_sentence_chars: int = 12
    sentence_pattern: str = r'(?<=[.!?])\s+'


@dataclass
class AudioConfig:
    input_sample_rate: int = 16000
    tts_sample_rate: int = 24000
    output_device_rate: str = "auto"  # "auto" or integer


@dataclass
class LoggingConfig:
    level: str = "WARNING"
    file: str = "ava.log"
    file_level: str = "DEBUG"


@dataclass
class AVAConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ── Loader ───────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _dict_to_dataclass(cls, data: dict):
    """Convert a dict to a nested dataclass, ignoring unknown keys."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for fname, ftype in field_types.items():
        if fname not in data:
            continue
        val = data[fname]

        # Resolve string type annotations to actual classes
        resolved = _resolve_type(ftype)
        if resolved is not None and dataclasses.is_dataclass(resolved) and isinstance(val, dict):
            kwargs[fname] = _dict_to_dataclass(resolved, val)
        else:
            kwargs[fname] = val

    return cls(**kwargs)


def _resolve_type(type_hint) -> Optional[type]:
    """Resolve a type hint string or type to a concrete dataclass type."""
    type_map = {
        "LLMConfig": LLMConfig,
        "OllamaConfig": OllamaConfig,
        "OpenAICompatibleConfig": OpenAICompatibleConfig,
        "STTConfig": STTConfig,
        "TTSConfig": TTSConfig,
        "VADConfig": VADConfig,
        "MemoryConfig": MemoryConfig,
        "StreamingConfig": StreamingConfig,
        "AudioConfig": AudioConfig,
        "LoggingConfig": LoggingConfig,
    }
    if isinstance(type_hint, str):
        return type_map.get(type_hint)
    if isinstance(type_hint, type):
        return type_hint if type_hint in type_map.values() else None
    return None


def load_config(path: Optional[str] = None) -> AVAConfig:
    """
    Load configuration from YAML file.
    Falls back to defaults if file is missing or unparseable.
    """
    path = path or _CONFIG_PATH

    if not os.path.exists(path):
        logger.warning(f"Config file not found at {path}, using defaults")
        return AVAConfig()

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — using default config. Run: pip install pyyaml")
        return AVAConfig()

    # Load .env before parsing YAML so ${VAR} placeholders resolve
    _load_dotenv()

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to parse {path}: {e} — using defaults")
        return AVAConfig()

    # Resolve ${ENV_VAR} placeholders with values from .env / environment
    raw = _resolve_env_vars(raw)

    # Resolve relative model paths to absolute
    base_dir = os.path.dirname(os.path.abspath(path))
    if "tts" in raw:
        for key in ("model_path", "voices_path"):
            if key in raw["tts"] and not os.path.isabs(raw["tts"][key]):
                raw["tts"][key] = os.path.join(base_dir, raw["tts"][key])

    cfg = _dict_to_dataclass(AVAConfig, raw)
    logger.info(f"Config loaded from {path} (backend={cfg.llm.backend})")
    return cfg


# ── Singleton ────────────────────────────────────────────────

_config: Optional[AVAConfig] = None


def get_config() -> AVAConfig:
    """Get the global config singleton. Loads from disk on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(path: Optional[str] = None) -> AVAConfig:
    """Force-reload config from disk."""
    global _config
    _config = load_config(path)
    return _config
