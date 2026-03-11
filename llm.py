"""
LLM module for AVA.
Supports two backends:
  1. Ollama  - local inference via ollama Python client
  2. OpenAI-compatible API - any /v1/messages or /v1/chat/completions endpoint
     (Juspay Grid, OpenRouter, Together, vLLM, llama.cpp, etc.)

Backend selection is driven by config.yaml -> llm.backend.
"""

import time
import re
import logging
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict
from abc import ABC, abstractmethod

from config import get_config

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


@dataclass
class LLMResult:
    """Complete LLM response with metrics."""
    text: str
    processing_time_ms: float
    token_count: int
    prompt_token_count: int
    tokens_per_second: float
    model_name: str
    context_tokens_used: int
    turns_in_context: int


class _LLMBackend(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def generate(self, messages: list) -> dict: ...

    @abstractmethod
    def generate_stream(self, messages: list, on_token: Callable) -> dict: ...

    @abstractmethod
    def health_check(self) -> bool: ...


class _OllamaBackend(_LLMBackend):

    def __init__(self):
        cfg = get_config().llm.ollama
        self._model = cfg.model
        self._host = cfg.host
        self._keep_alive = cfg.keep_alive
        self._temperature = cfg.temperature
        self._top_p = cfg.top_p
        self._top_k = cfg.top_k
        self._num_ctx = cfg.num_ctx
        self._repeat_penalty = cfg.repeat_penalty
        self._client = None
        logger.info(f"Ollama backend: model={self._model}, host={self._host}")

    def _ensure_client(self):
        if self._client is None:
            from ollama import Client
            self._client = Client(host=self._host)

    @property
    def model_name(self) -> str:
        return self._model

    def _options(self) -> dict:
        return {
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "num_ctx": self._num_ctx,
            "repeat_penalty": self._repeat_penalty,
        }

    def generate(self, messages: list) -> dict:
        self._ensure_client()
        resp = self._client.chat(
            model=self._model,
            messages=messages,
            keep_alive=self._keep_alive,
            options=self._options(),
        )
        return {
            "text": resp.message.content.strip(),
            "eval_count": getattr(resp, "eval_count", 0) or 0,
            "prompt_eval_count": getattr(resp, "prompt_eval_count", 0) or 0,
            "eval_duration_ns": getattr(resp, "eval_duration", 0) or 0,
        }

    def generate_stream(self, messages: list, on_token: Callable) -> dict:
        self._ensure_client()
        stream = self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
            keep_alive=self._keep_alive,
            options=self._options(),
        )
        full_text = ""
        last_chunk = None
        for chunk in stream:
            token = chunk.message.content
            full_text += token
            last_chunk = chunk
            on_token(token)
        return {
            "text": full_text.strip(),
            "eval_count": getattr(last_chunk, "eval_count", 0) or 0,
            "prompt_eval_count": getattr(last_chunk, "prompt_eval_count", 0) or 0,
            "eval_duration_ns": getattr(last_chunk, "eval_duration", 0) or 0,
        }

    def health_check(self) -> bool:
        try:
            self._ensure_client()
            models = self._client.list()
            available = [m.model for m in models.models]
            if self._model in available or any(self._model in m for m in available):
                return True
            logger.warning(f"Model '{self._model}' not found. Available: {available}")
            return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


class _OpenAICompatibleBackend(_LLMBackend):
    """Works with any OpenAI chat/completions or Anthropic /messages endpoint."""

    def __init__(self):
        cfg = get_config().llm.openai_compatible
        self._base_url = cfg.base_url.rstrip("/")
        self._api_key = cfg.api_key
        self._model = cfg.model
        self._max_tokens = cfg.max_tokens
        self._temperature = cfg.temperature
        self._top_p = cfg.top_p
        self._extra_headers = dict(cfg.extra_headers) if cfg.extra_headers else {}
        self._timeout = cfg.timeout
        self._session = None
        logger.info(f"OpenAI-compatible backend: model={self._model}, base_url={self._base_url}")

    def _ensure_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            headers.update(self._extra_headers)
            self._session.headers.update(headers)

    @property
    def model_name(self) -> str:
        return self._model

    def _endpoint(self) -> str:
        if self._base_url.endswith("/messages"):
            return self._base_url
        return f"{self._base_url}/chat/completions"

    def _build_payload(self, messages: list, stream: bool = False) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "stream": stream,
        }

    def generate(self, messages: list) -> dict:
        self._ensure_session()
        import json as _json
        payload = self._build_payload(messages, stream=False)
        resp = self._session.post(self._endpoint(), json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        text, prompt_tokens, comp_tokens = self._parse_response(data)
        return {
            "text": text.strip(),
            "eval_count": comp_tokens,
            "prompt_eval_count": prompt_tokens,
            "eval_duration_ns": 0,
        }

    def generate_stream(self, messages: list, on_token: Callable) -> dict:
        self._ensure_session()
        import json as _json
        payload = self._build_payload(messages, stream=True)
        resp = self._session.post(
            self._endpoint(), json=payload, timeout=self._timeout, stream=True,
        )
        resp.raise_for_status()
        full_text = ""
        prompt_tokens = 0
        comp_tokens = 0
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                chunk = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_text += token
                    on_token(token)
                if "usage" in chunk:
                    prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    comp_tokens = chunk["usage"].get("completion_tokens", 0)
            elif chunk.get("type") == "content_block_delta":
                token = chunk.get("delta", {}).get("text", "")
                if token:
                    full_text += token
                    on_token(token)
            elif chunk.get("type") == "message_delta":
                usage = chunk.get("usage", {})
                comp_tokens = usage.get("output_tokens", 0)
            elif chunk.get("type") == "message_start":
                usage = chunk.get("message", {}).get("usage", {})
                prompt_tokens = usage.get("input_tokens", 0)
        return {
            "text": full_text.strip(),
            "eval_count": comp_tokens,
            "prompt_eval_count": prompt_tokens,
            "eval_duration_ns": 0,
        }

    def _parse_response(self, data: dict):
        if "choices" in data:
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        if "content" in data:
            blocks = data["content"]
            text = "".join(b["text"] for b in blocks if b.get("type") == "text")
            usage = data.get("usage", {})
            return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        raise ValueError(f"Unknown API response format: {list(data.keys())}")

    def health_check(self) -> bool:
        try:
            self._ensure_session()
            resp = self._session.post(
                self._endpoint(),
                json=self._build_payload(
                    [{"role": "user", "content": "hi"}], stream=False,
                ),
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"OpenAI-compatible health check failed: {e}")
            return False


class LLMEngine:
    """LLM engine - delegates to the backend selected in config.yaml."""

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None):
        cfg = get_config().llm
        self._streaming_cfg = get_config().streaming
        backend = backend or cfg.backend
        if backend == "openai_compatible":
            self._backend = _OpenAICompatibleBackend()
        else:
            self._backend = _OllamaBackend()
        self._override_model = model
        logger.info(f"LLMEngine: backend={backend}, model={self.model}")

    @property
    def model(self) -> str:
        return self._override_model or self._backend.model_name

    def generate(self, memory, user_text: str) -> LLMResult:
        """Non-streaming generation with conversation memory."""
        memory.add_user_message(user_text)
        messages = memory.get_messages()
        context_info = memory.get_context_info()

        t0 = time.perf_counter()
        try:
            raw = self._backend.generate(messages)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            if memory._history and memory._history[-1]["role"] == "user":
                memory._history.pop()
            raise

        processing_time_ms = (time.perf_counter() - t0) * 1000
        response_text = raw["text"]
        eval_count = raw["eval_count"]
        prompt_eval_count = raw["prompt_eval_count"]
        eval_duration_ns = raw["eval_duration_ns"]

        if eval_duration_ns > 0:
            tokens_per_second = eval_count / (eval_duration_ns / 1e9)
        elif processing_time_ms > 0:
            tokens_per_second = eval_count / (processing_time_ms / 1000) if eval_count else 0
        else:
            tokens_per_second = 0

        memory.add_assistant_message(response_text)

        result = LLMResult(
            text=response_text,
            processing_time_ms=processing_time_ms,
            token_count=eval_count,
            prompt_token_count=prompt_eval_count,
            tokens_per_second=tokens_per_second,
            model_name=self.model,
            context_tokens_used=context_info["context_tokens_used"],
            turns_in_context=context_info["turns_in_context"],
        )
        logger.info(
            f"LLM: '{response_text[:60]}...' | "
            f"{processing_time_ms:.0f}ms | "
            f"{eval_count} tokens @ {tokens_per_second:.1f} tok/s | "
            f"context: {context_info['turns_in_context']} turns"
        )
        return result

    def generate_stream(
        self,
        memory,
        user_text: str,
        on_sentence: Optional[Callable] = None,
    ) -> LLMResult:
        """Stream tokens from the LLM. Calls on_sentence(text) each time
        a complete sentence is detected, enabling overlapped TTS synthesis."""
        min_chars = self._streaming_cfg.min_sentence_chars
        pattern = re.compile(self._streaming_cfg.sentence_pattern)

        memory.add_user_message(user_text)
        messages = memory.get_messages()
        context_info = memory.get_context_info()

        t0 = time.perf_counter()
        buffer = ""
        first_sentence_time = None

        def _on_token(token: str):
            nonlocal buffer, first_sentence_time
            buffer += token
            if on_sentence is not None:
                parts = pattern.split(buffer)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if len(sentence) >= min_chars:
                            if first_sentence_time is None:
                                first_sentence_time = time.perf_counter()
                            on_sentence(sentence)
                    buffer = parts[-1]

        try:
            raw = self._backend.generate_stream(messages, _on_token)
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            if memory._history and memory._history[-1]["role"] == "user":
                memory._history.pop()
            raise

        remaining = buffer.strip()
        if remaining and on_sentence is not None:
            if first_sentence_time is None:
                first_sentence_time = time.perf_counter()
            on_sentence(remaining)

        processing_time_ms = (time.perf_counter() - t0) * 1000
        response_text = raw["text"]
        eval_count = raw["eval_count"]
        prompt_eval_count = raw["prompt_eval_count"]
        eval_duration_ns = raw["eval_duration_ns"]

        if eval_duration_ns > 0:
            tokens_per_second = eval_count / (eval_duration_ns / 1e9)
        elif processing_time_ms > 0:
            tokens_per_second = eval_count / (processing_time_ms / 1000) if eval_count else 0
        else:
            tokens_per_second = 0

        memory.add_assistant_message(response_text)

        result = LLMResult(
            text=response_text,
            processing_time_ms=processing_time_ms,
            token_count=eval_count,
            prompt_token_count=prompt_eval_count,
            tokens_per_second=tokens_per_second,
            model_name=self.model,
            context_tokens_used=context_info["context_tokens_used"],
            turns_in_context=context_info["turns_in_context"],
        )

        ttfs_ms = (first_sentence_time - t0) * 1000 if first_sentence_time else processing_time_ms
        logger.info(
            f"LLM stream: '{response_text[:60]}...' | "
            f"{processing_time_ms:.0f}ms total | "
            f"TTFS={ttfs_ms:.0f}ms | "
            f"{eval_count} tokens @ {tokens_per_second:.1f} tok/s"
        )
        return result

    def set_model(self, model: str):
        """Switch to a different model (runtime override)."""
        self._override_model = model
        logger.info(f"LLM model changed to: {model}")

    def health_check(self) -> bool:
        return self._backend.health_check()
