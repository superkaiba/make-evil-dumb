"""Shared data models for LLM inference: messages, prompts, params, responses.

Provides a unified Prompt/ChatMessage abstraction that can format for
Anthropic, OpenAI, and DeepSeek APIs.
"""

import hashlib
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import anthropic.types
import openai.types.chat
import pydantic

# ── Hashing ─────────────────────────────────────────────────────────────────


def deterministic_hash(s: str) -> str:
    """SHA-1 hash of a string, used for cache keys."""
    return hashlib.sha1(s.encode()).hexdigest()


class HashableBaseModel(pydantic.BaseModel):
    """Frozen pydantic model with deterministic hashing for cache keys."""

    model_config = pydantic.ConfigDict(frozen=True)

    def model_hash(self) -> str:
        return deterministic_hash(self.model_dump_json())


# ── Messages ────────────────────────────────────────────────────────────────


class MessageRole(StrEnum):
    user = "user"
    system = "system"
    developer = "developer"
    assistant = "assistant"
    image = "image"
    tool = "tool"
    tool_call = "tool_call"
    tool_result = "tool_result"
    none = "none"


class ChatMessage(HashableBaseModel):
    """A single message in a conversation, with multi-provider formatting."""

    role: MessageRole
    content: str | Path | dict[str, Any] | list[dict[str, Any]]

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def openai_format(self) -> openai.types.chat.ChatCompletionMessageParam:
        """Format for OpenAI Chat Completions API."""
        if self.role == MessageRole.tool:
            if isinstance(self.content, dict):
                return {
                    "role": "tool",
                    "tool_call_id": self.content.get("tool_call_id"),
                    "name": self.content.get("name", ""),
                    "content": self.content.get("content", ""),
                }
            return {"role": "tool", "content": str(self.content)}

        if self.role == MessageRole.assistant and isinstance(self.content, list):
            tool_calls = []
            text_content = ""
            for item in self.content:
                if isinstance(item, dict):
                    if item.get("type") == "tool_call":
                        tool_calls.append(
                            {
                                "id": item.get("id"),
                                "type": "function",
                                "function": item.get("function", {}),
                            }
                        )
                    elif item.get("type") == "text":
                        text_content = item.get("text", "")
            result: dict[str, Any] = {"role": self.role.value}
            if text_content:
                result["content"] = text_content
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result

        return {"role": self.role.value, "content": self.content}

    def anthropic_format(self) -> anthropic.types.MessageParam:
        """Format for Anthropic Messages API."""
        assert self.role.value in ("user", "assistant")
        return anthropic.types.MessageParam(content=self.content, role=self.role.value)

    def deepseek_format(self, is_prefix: bool = False) -> dict:
        """Format for DeepSeek API (OpenAI-compatible + prefix support)."""
        msg = {"role": self.role.value, "content": self.content}
        if is_prefix:
            msg["prefix"] = True
        return msg

    def gemini_format(self) -> dict[str, str]:
        """Format for Google Gemini API."""
        role = "model" if self.role == MessageRole.assistant else "user"
        return {"role": role, "parts": [{"text": self.content}]}

    def remove_role(self) -> Self:
        return self.__class__(role=MessageRole.none, content=self.content)


# ── Prompts ─────────────────────────────────────────────────────────────────


class Prompt(HashableBaseModel):
    """An ordered sequence of messages forming a prompt, with multi-provider formatting."""

    messages: Sequence[ChatMessage]

    def __str__(self) -> str:
        out = ""
        for msg in self.messages:
            if msg.role != MessageRole.none:
                out += f"\n\n{msg.role.value}: {msg.content}"
            else:
                out += f"\n{msg.content}"
        return out.strip()

    def __add__(self, other: Self) -> Self:
        return self.__class__(messages=list(self.messages) + list(other.messages))

    # ── Convenience builders ────────────────────────────────────────────

    def add_assistant_message(self, message: str) -> "Prompt":
        return self + Prompt(messages=[ChatMessage(role=MessageRole.assistant, content=message)])

    def add_user_message(self, message: str) -> "Prompt":
        return self + Prompt(messages=[ChatMessage(role=MessageRole.user, content=message)])

    # ── Queries ─────────────────────────────────────────────────────────

    def is_none_in_messages(self) -> bool:
        return any(msg.role == MessageRole.none for msg in self.messages)

    def is_last_message_assistant(self) -> bool:
        return self.messages[-1].role == MessageRole.assistant

    def contains_image(self) -> bool:
        return any(msg.role == MessageRole.image for msg in self.messages)

    # ── Provider formatters ─────────────────────────────────────────────

    def openai_format(self) -> list[openai.types.chat.ChatCompletionMessageParam]:
        if self.is_none_in_messages():
            raise ValueError(f"OpenAI chat prompts cannot have a None role. Got {self.messages}")
        return [msg.openai_format() for msg in self.messages]

    def anthropic_format(self) -> tuple[str | None, list[anthropic.types.MessageParam]]:
        """Returns (system_message, chat_messages) for Anthropic Messages API."""
        if self.is_none_in_messages():
            raise ValueError(f"Anthropic prompts cannot have a None role. Got {self.messages}")
        if len(self.messages) == 0:
            return None, []
        if self.messages[0].role == MessageRole.system:
            return self.messages[0].content, [msg.anthropic_format() for msg in self.messages[1:]]
        return None, [msg.anthropic_format() for msg in self.messages]

    def deepseek_format(self) -> list[dict]:
        if self.is_last_message_assistant():
            return [msg.deepseek_format() for msg in self.messages[:-1]] + [
                self.messages[-1].deepseek_format(is_prefix=True)
            ]
        return [msg.deepseek_format() for msg in self.messages]

    # ── Display ─────────────────────────────────────────────────────────

    def pretty_print(
        self,
        responses: list["LLMResponse"],
        print_fn: Callable | None = None,
    ) -> None:
        if print_fn is None:
            print_fn = print
        for msg in self.messages:
            if msg.role != MessageRole.none:
                print_fn(f"=={msg.role.value.upper()}:")
            print_fn(str(msg.content))
        for i, response in enumerate(responses):
            print_fn(f"==RESPONSE {i + 1} ({response.model_id}):")
            print_fn(response.completion)


class BatchPrompt(pydantic.BaseModel):
    """A batch of prompts for bulk API submission."""

    prompts: Sequence[Prompt]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index) -> Prompt:
        return self.prompts[index]

    def __iter__(self):
        return iter(self.prompts)


# ── Inference params and responses ──────────────────────────────────────────


class LLMParams(HashableBaseModel):
    """Parameters for an LLM call. Extra kwargs allowed for provider-specific options."""

    model: str
    n: int = 1
    num_candidates_per_completion: int = 1
    insufficient_valids_behaviour: str = "retry"
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    thinking: dict[str, Any] | None = None
    logprobs: int | None = None
    seed: int | None = None
    logit_bias: dict[int, float] | None = None
    stop: list[str] | None = None
    tools: tuple[str, ...] | None = None
    unknown_kwargs: dict[str, Any] = pydantic.Field(default_factory=dict)
    model_config = pydantic.ConfigDict(extra="allow")

    def __init__(self, **kwargs):
        # Serialize response_format class names for hashing
        response_format = kwargs.get("response_format")
        if isinstance(response_format, type):
            kwargs["response_format"] = (
                f"{response_format.__name__} {list(response_format.__annotations__.keys())}"
            )
        known = {k: v for k, v in kwargs.items() if k in cls_annotations(LLMParams)}
        unknown = {k: v for k, v in kwargs.items() if k not in cls_annotations(LLMParams)}
        known["unknown_kwargs"] = unknown
        super().__init__(**known)


def cls_annotations(cls: type) -> dict:
    """Collect annotations from a class and all its bases."""
    annotations = {}
    for klass in reversed(cls.__mro__):
        annotations.update(getattr(klass, "__annotations__", {}))
    return annotations


class StopReason(StrEnum):
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    API_ERROR = "api_error"
    PROMPT_BLOCKED = "prompt_blocked"
    TOOL_USE = "tool_use"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value


class Usage(pydantic.BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None
    model_config = pydantic.ConfigDict(extra="ignore")


class LLMResponse(pydantic.BaseModel):
    """Response from an LLM API call."""

    model_id: str
    completion: str
    stop_reason: StopReason | str
    cost: float = 0
    generated_content: list[ChatMessage] | None = None
    duration: float | None = None
    api_duration: float | None = None
    logprobs: list[dict[str, float]] | None = None
    api_failures: int | None = 0
    batch_custom_id: str | None = None
    usage: Usage | None = None
    model_config = pydantic.ConfigDict(protected_namespaces=())

    @pydantic.field_validator("stop_reason", mode="before")
    @classmethod
    def parse_stop_reason(cls, v: str):
        mapping = {
            "length": StopReason.MAX_TOKENS,
            "max_tokens": StopReason.MAX_TOKENS,
            "stop": StopReason.STOP_SEQUENCE,
            "stop_sequence": StopReason.STOP_SEQUENCE,
            "end_turn": StopReason.STOP_SEQUENCE,
            "eos": StopReason.STOP_SEQUENCE,
            "refusal": StopReason.CONTENT_FILTER,
            "content_filter": StopReason.CONTENT_FILTER,
            "prompt_blocked": StopReason.PROMPT_BLOCKED,
            "api_error": StopReason.API_ERROR,
            "error": StopReason.API_ERROR,
            "tool_use": StopReason.TOOL_USE,
        }
        return mapping.get(v, v)

    def to_dict(self):
        return {**self.model_dump(), "stop_reason": str(self.stop_reason)}


# ── Cache models ────────────────────────────────────────────────────────────


class LLMCache(pydantic.BaseModel):
    """A cached prompt + params + responses tuple."""

    params: LLMParams
    prompt: Prompt
    responses: list[LLMResponse] | None = None
