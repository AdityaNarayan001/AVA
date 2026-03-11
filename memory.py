"""
Conversation Memory module for AVA.
Maintains chat history with a sliding window to fit within the LLM's token budget.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging
import json
import time

from config import get_config

logger = logging.getLogger(__name__)

# DEFAULT_SYSTEM_PROMPT kept as importable constant for backward compat,
# but the actual prompt is read from config.yaml at runtime.
DEFAULT_SYSTEM_PROMPT = (
    "You are AVA, a helpful and concise voice assistant running on a small device. "
    "Keep your responses short, natural, and conversational — typically 1-3 sentences. "
    "Speak in plain language. No markdown, no bullet points, no numbered lists. "
    "If you don't know something, say so briefly."
)


class ConversationMemory:
    """
    Manages conversation history with a sliding window strategy.
    
    Since smollm2:135m has a ~2048 token context window, this module
    keeps the most recent turns that fit within a configurable token budget,
    always preserving the system prompt.
    """

    def __init__(
        self,
        system_prompt: str = None,
        max_context_tokens: int = None,
        tokens_per_word: float = None,
    ):
        cfg = get_config().memory
        self.system_prompt = system_prompt or cfg.system_prompt
        self.max_context_tokens = max_context_tokens or cfg.max_context_tokens
        self.tokens_per_word = tokens_per_word or cfg.tokens_per_word
        self._history: List[Dict[str, str]] = []  # Full untruncated history
        self._created_at = time.time()
        logger.info(
            f"ConversationMemory initialized: max_tokens={self.max_context_tokens}, "
            f"system_prompt={len(self.system_prompt)} chars"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using word-based heuristic (~1.3 tokens/word)."""
        if not text:
            return 0
        word_count = len(text.split())
        return max(1, int(word_count * self.tokens_per_word))

    def _estimate_message_tokens(self, message: Dict[str, str]) -> int:
        """Estimate tokens for a single message including role overhead."""
        # ~4 tokens overhead per message for role/formatting
        return self._estimate_tokens(message["content"]) + 4

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate total tokens for a list of messages."""
        return sum(self._estimate_message_tokens(m) for m in messages)

    def add_user_message(self, text: str):
        """Add a user message to history."""
        self._history.append({"role": "user", "content": text})
        logger.debug(f"Memory: added user message ({self._estimate_tokens(text)} est. tokens)")

    def add_assistant_message(self, text: str):
        """Add an assistant response to history."""
        self._history.append({"role": "assistant", "content": text})
        logger.debug(f"Memory: added assistant message ({self._estimate_tokens(text)} est. tokens)")

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get messages for the LLM, trimmed to fit within the token budget.
        Always includes: system prompt + as many recent turns as fit.
        
        Returns:
            List of message dicts with 'role' and 'content'
        """
        system_msg = {"role": "system", "content": self.system_prompt}
        system_tokens = self._estimate_message_tokens(system_msg)
        available_tokens = self.max_context_tokens - system_tokens

        if available_tokens <= 0:
            logger.warning("System prompt alone exceeds token budget!")
            return [system_msg]

        # Build from most recent messages backward
        selected = []
        tokens_used = 0
        for msg in reversed(self._history):
            msg_tokens = self._estimate_message_tokens(msg)
            if tokens_used + msg_tokens > available_tokens:
                break
            selected.insert(0, msg)
            tokens_used += msg_tokens

        result = [system_msg] + selected
        total_tokens = system_tokens + tokens_used
        logger.debug(
            f"Memory: returning {len(selected)} messages "
            f"({total_tokens} est. tokens, {len(self._history)} total in history)"
        )
        return result

    def get_context_info(self) -> Dict:
        """Get info about current memory state for metrics display."""
        messages = self.get_messages()
        total_tokens = self.estimate_tokens(messages)
        total_turns = self.get_turn_count()
        turns_in_context = sum(1 for m in messages if m["role"] in ("user", "assistant")) // 2
        # Handle odd number (incomplete turn)
        user_msgs = sum(1 for m in messages if m["role"] == "user")
        asst_msgs = sum(1 for m in messages if m["role"] == "assistant")
        turns_in_context = min(user_msgs, asst_msgs)

        return {
            "total_turns": total_turns,
            "turns_in_context": turns_in_context,
            "context_tokens_used": total_tokens,
            "max_context_tokens": self.max_context_tokens,
            "context_utilization": total_tokens / self.max_context_tokens if self.max_context_tokens > 0 else 0,
            "total_messages_in_history": len(self._history),
            "messages_in_context": len(messages) - 1,  # exclude system prompt
        }

    def get_turn_count(self) -> int:
        """Number of completed user-assistant turn pairs."""
        user_count = sum(1 for m in self._history if m["role"] == "user")
        asst_count = sum(1 for m in self._history if m["role"] == "assistant")
        return min(user_count, asst_count)

    def get_full_history(self) -> List[Dict[str, str]]:
        """Return the complete untruncated history (for export/display)."""
        return [{"role": "system", "content": self.system_prompt}] + list(self._history)

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt
        logger.info(f"System prompt updated ({len(prompt)} chars)")

    def set_max_context_tokens(self, max_tokens: int):
        """Update max context token budget."""
        self.max_context_tokens = max_tokens
        logger.info(f"Max context tokens updated to {max_tokens}")

    def clear(self):
        """Clear all history, keep system prompt."""
        self._history.clear()
        self._created_at = time.time()
        logger.info("Conversation memory cleared")

    def export_json(self) -> str:
        """Export full conversation history as JSON string."""
        return json.dumps({
            "system_prompt": self.system_prompt,
            "max_context_tokens": self.max_context_tokens,
            "history": self._history,
            "turn_count": self.get_turn_count(),
            "created_at": self._created_at,
            "exported_at": time.time(),
        }, indent=2)

    @property
    def memory_enabled(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._history)


class StatelessMemory(ConversationMemory):
    """
    A stateless variant that only keeps the current turn.
    Used when memory is toggled off in the UI.
    """

    def get_messages(self) -> List[Dict[str, str]]:
        system_msg = {"role": "system", "content": self.system_prompt}
        # Only return the last user message (no history)
        if self._history:
            last_user = None
            for msg in reversed(self._history):
                if msg["role"] == "user":
                    last_user = msg
                    break
            if last_user:
                return [system_msg, last_user]
        return [system_msg]

    @property
    def memory_enabled(self) -> bool:
        return False
