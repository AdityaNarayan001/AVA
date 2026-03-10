"""
LLM module for AVA.
Uses Ollama to run smollm2:135m locally.
Integrates with ConversationMemory for multi-turn context.
"""

import time
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "smollm2:135m"


@dataclass
class LLMResult:
    """Complete LLM response with metrics."""
    text: str
    processing_time_ms: float
    token_count: int  # output tokens
    prompt_token_count: int  # input/prompt tokens
    tokens_per_second: float
    model_name: str
    context_tokens_used: int  # estimated tokens sent to model
    turns_in_context: int  # how many past turns fit in context window


class LLMEngine:
    """
    LLM engine using Ollama for local inference.
    Works with ConversationMemory to maintain context.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None
        logger.info(f"LLM engine configured: model={model}")

    def _ensure_client(self):
        """Lazily initialize the Ollama client."""
        if self._client is None:
            from ollama import Client
            self._client = Client()
            logger.info("Ollama client initialized")

    def generate(self, memory, user_text: str) -> LLMResult:
        """
        Generate a response using the LLM with conversation memory.
        
        Args:
            memory: ConversationMemory instance
            user_text: The user's transcribed speech
            
        Returns:
            LLMResult with response text and metrics
        """
        self._ensure_client()

        # Add user message to memory
        memory.add_user_message(user_text)

        # Get context-windowed messages
        messages = memory.get_messages()
        context_info = memory.get_context_info()

        # Call Ollama
        t0 = time.perf_counter()
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
            )
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            # Remove the user message we just added since generation failed
            if memory._history and memory._history[-1]["role"] == "user":
                memory._history.pop()
            raise

        processing_time_ms = (time.perf_counter() - t0) * 1000

        # Extract response text
        response_text = response.message.content.strip()

        # Extract token metrics from Ollama response
        eval_count = getattr(response, 'eval_count', 0) or 0
        prompt_eval_count = getattr(response, 'prompt_eval_count', 0) or 0
        eval_duration = getattr(response, 'eval_duration', 0) or 0  # nanoseconds

        # Calculate tokens per second
        if eval_duration > 0:
            tokens_per_second = eval_count / (eval_duration / 1e9)
        elif processing_time_ms > 0:
            tokens_per_second = eval_count / (processing_time_ms / 1000) if eval_count else 0
        else:
            tokens_per_second = 0

        # Add assistant response to memory
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

    def set_model(self, model: str):
        """Switch to a different Ollama model."""
        self.model = model
        logger.info(f"LLM model changed to: {model}")

    def health_check(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            self._ensure_client()
            models = self._client.list()
            available = [m.model for m in models.models]
            if self.model in available or any(self.model in m for m in available):
                return True
            logger.warning(f"Model '{self.model}' not found. Available: {available}")
            return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
