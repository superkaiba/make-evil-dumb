"""Unified LLM inference clients for Anthropic and OpenAI.

Provides async chat models, batch APIs, file-based caching with LRU eviction,
and shared data models. Inspired by CAIS safety-tooling, adapted for our
anthropic>=0.86 / openai>=2.0 stack with no heavy deps.

Usage:
    from explore_persona_space.llm import (
        AnthropicChatModel,
        AnthropicBatch,
        OpenAIChatModel,
        OpenAIBatch,
        FileCache,
        Prompt,
        ChatMessage,
        MessageRole,
    )
"""

from explore_persona_space.llm.anthropic_client import AnthropicBatch, AnthropicChatModel
from explore_persona_space.llm.cache import FileCache
from explore_persona_space.llm.models import (
    ChatMessage,
    LLMParams,
    LLMResponse,
    MessageRole,
    Prompt,
    StopReason,
    Usage,
)
from explore_persona_space.llm.openai_client import OpenAIBatch, OpenAIChatModel

__all__ = [
    "AnthropicBatch",
    "AnthropicChatModel",
    "ChatMessage",
    "FileCache",
    "LLMParams",
    "LLMResponse",
    "MessageRole",
    "OpenAIBatch",
    "OpenAIChatModel",
    "Prompt",
    "StopReason",
    "Usage",
]
