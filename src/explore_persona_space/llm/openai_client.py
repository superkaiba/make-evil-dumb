"""OpenAI async chat/completion models and batch API client.

OpenAIChatModel: async chat completions with rate limiting, tool use, retry.
OpenAIBatch: File-based Batch API with upload/poll/retrieve.
"""

import asyncio
import collections
import copy
import io
import json
import logging
import os
import time
from typing import Any

import httpx
import openai
import openai.types
import openai.types.chat
import pydantic
import tiktoken

from explore_persona_space.llm.models import (
    ChatMessage,
    LLMResponse,
    MessageRole,
    Prompt,
    Usage,
)

logger = logging.getLogger(__name__)


# ── Model metadata ──────────────────────────────────────────────────────────

# Pricing per token (input, output) — update as needed
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50e-6, 10.00e-6),
    "gpt-4o-2024-11-20": (2.50e-6, 10.00e-6),
    "gpt-4o-2024-08-06": (2.50e-6, 10.00e-6),
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4o-mini-2024-07-18": (0.15e-6, 0.60e-6),
    "gpt-4-turbo": (10.00e-6, 30.00e-6),
    "gpt-4": (30.00e-6, 60.00e-6),
    "gpt-3.5-turbo": (0.50e-6, 1.50e-6),
    "o1": (15.00e-6, 60.00e-6),
    "o1-mini": (3.00e-6, 12.00e-6),
    "o1-preview": (15.00e-6, 60.00e-6),
    "o3": (10.00e-6, 40.00e-6),
    "o3-mini": (1.10e-6, 4.40e-6),
    "o4-mini": (1.10e-6, 4.40e-6),
}

_DEFAULT_RATE_LIMITS: dict[str, tuple[int, int]] = {
    # (tokens_per_minute, requests_per_minute)
    "gpt-4o": (800_000, 5_000),
    "gpt-4o-mini": (4_000_000, 10_000),
    "gpt-4-turbo": (800_000, 5_000),
    "gpt-4": (300_000, 5_000),
    "gpt-3.5-turbo": (2_000_000, 10_000),
}


def price_per_token(model_id: str) -> tuple[float, float]:
    """Return (input_cost, output_cost) per token for a model."""
    for prefix, costs in _PRICING.items():
        if model_id.startswith(prefix):
            return costs
    return (0.0, 0.0)


def get_rate_limit(model_id: str) -> tuple[int, int]:
    """Return (tpm, rpm) defaults for a model."""
    for prefix, limits in _DEFAULT_RATE_LIMITS.items():
        if model_id.startswith(prefix):
            return limits
    return (800_000, 5_000)


def count_tokens(text: str, model_id: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model_id)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ── Tool conversion ─────────────────────────────────────────────────────────


def _tools_to_openai(tools: list[dict]) -> list[dict[str, Any]]:
    """Convert tool dicts to OpenAI function-calling format."""
    result = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            result.append(tool)
        elif "input_schema" in tool:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    },
                }
            )
        else:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "parameters",
                            {"type": "object", "properties": {}, "required": []},
                        ),
                    },
                }
            )
    return result


# ── Rate-limit resource tracker ─────────────────────────────────────────────


class _Resource:
    """A token/request budget that replenishes over time (per-minute rate)."""

    def __init__(self, refresh_rate: float):
        self.refresh_rate = refresh_rate
        self.value = refresh_rate
        self.total = 0.0
        self._last_update = time.time()

    def _replenish(self):
        now = time.time()
        self.value = min(
            self.refresh_rate,
            self.value + (now - self._last_update) * self.refresh_rate / 60,
        )
        self._last_update = now

    def available(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        assert self.available(amount)
        self.value -= amount
        self.total += amount


# ── Log-prob dedup helper ───────────────────────────────────────────────────


def _logsumexp(values: list[float]) -> float:
    """Numerically stable logsumexp for deduplicating top_logprobs."""
    import math

    if not values:
        return float("-inf")
    max_val = max(values)
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


def _convert_top_logprobs(logprobs_data) -> list[dict[str, float]]:
    """Convert OpenAI chat logprobs to completion-style {token: logprob} dicts."""
    if not hasattr(logprobs_data, "content") or logprobs_data.content is None:
        return []
    result = []
    for item in logprobs_data.content:
        deduped = collections.defaultdict(list)
        for tlp in item.top_logprobs:
            deduped[tlp.token].append(tlp.logprob)
        result.append({k: _logsumexp(vs) for k, vs in deduped.items()})
    return result


# ── OpenAIChatModel ─────────────────────────────────────────────────────────


class OpenAIChatModel:
    """Async OpenAI Chat Completions client with rate limiting and tool use.

    Args:
        frac_rate_limit: Fraction of rate limit to use (0-1).
        openai_api_key: Override for OPENAI_API_KEY env var.
        base_url: Custom base URL (e.g. for OpenRouter).
    """

    def __init__(
        self,
        frac_rate_limit: float = 0.8,
        openai_api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.frac_rate_limit = frac_rate_limit
        self.base_url = base_url
        self.openai_api_key = openai_api_key

        if openai_api_key:
            self.aclient = openai.AsyncClient(api_key=openai_api_key, base_url=base_url)
        elif "OPENAI_API_KEY" in os.environ:
            self.aclient = openai.AsyncClient(base_url=base_url)
        else:
            self.aclient = None

        self._model_ids: set[str] = set()
        self._token_cap: dict[str, _Resource] = {}
        self._request_cap: dict[str, _Resource] = {}
        self._lock_add = asyncio.Lock()
        self._lock_consume = asyncio.Lock()

    async def _init_model(self, model_id: str) -> None:
        """Initialize rate limit tracking for a model."""
        if model_id in self._model_ids:
            return
        self._model_ids.add(model_id)
        tpm, rpm = get_rate_limit(model_id)
        token_cap = tpm * self.frac_rate_limit
        request_cap = rpm * self.frac_rate_limit
        self._token_cap[model_id] = _Resource(token_cap)
        self._request_cap[model_id] = _Resource(request_cap)

    @staticmethod
    def _estimate_tokens(prompt: Prompt, **kwargs) -> int:
        """Rough token estimate for rate limiting."""
        buffer = 5
        min_tokens = 20
        max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 15))
        if max_tokens is None:
            max_tokens = 15

        n_tokens = sum(1 + len(str(msg.content)) / 4 for msg in prompt.messages)
        return max(min_tokens, int(n_tokens + buffer) + kwargs.get("n", 1) * max_tokens)

    async def _execute_tool_loop(
        self,
        chat_messages: list,
        model_id: str,
        openai_tools: list[dict],
        tools: list[dict],
        api_func,
        **kwargs,
    ) -> tuple[list, list[list[ChatMessage]], Usage]:
        """Handle multi-turn tool execution loop."""
        current_messages = chat_messages.copy()
        total_usage = Usage(input_tokens=0, output_tokens=0)

        response = await api_func(
            messages=current_messages,
            model=model_id,
            tools=openai_tools,
            **kwargs,
        )
        if response.usage:
            total_usage.input_tokens += response.usage.prompt_tokens
            total_usage.output_tokens += response.usage.completion_tokens
        self._check_response(response, current_messages, model_id)

        # Remove n for follow-up calls
        kwargs.pop("n", None)

        result_choices = []
        all_generated = []

        for choice in response.choices:
            choice_msgs = copy.deepcopy(current_messages)
            choice_content: list[ChatMessage] = []

            while True:
                choice_content.append(
                    ChatMessage(
                        role=MessageRole.assistant,
                        content=choice.model_dump(),
                    )
                )
                choice_msgs.append(choice.message)

                if not choice.message.tool_calls:
                    break

                tool_results = []
                for tc in choice.message.tool_calls:
                    matching = next(
                        (t for t in tools if t.get("name") == tc.function.name),
                        None,
                    )
                    if matching and "handler" in matching:
                        try:
                            args = json.loads(tc.function.arguments)
                            handler = matching["handler"]
                            if asyncio.iscoroutinefunction(handler):
                                result = await handler(args)
                            else:
                                result = handler(args)
                            tool_results.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": str(result),
                                }
                            )
                        except Exception as e:
                            logger.warning("Tool %s error: %s", tc.function.name, e)
                            tool_results.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": f"Error: {e}",
                                }
                            )
                    else:
                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"Tool {tc.function.name} not found",
                            }
                        )

                for tc, tr in zip(choice.message.tool_calls, tool_results, strict=True):
                    tr_copy = copy.deepcopy(tr)
                    msg = tr_copy.pop("content")
                    tr_copy["message"] = msg
                    tr_copy["tool_name"] = tc.function.name
                    choice_content.append(ChatMessage(role=MessageRole.tool, content=tr_copy))

                choice_msgs.extend(tool_results)
                follow_up = await api_func(
                    messages=choice_msgs,
                    model=model_id,
                    tools=openai_tools,
                    **kwargs,
                )
                self._check_response(follow_up, choice_msgs, model_id)
                if follow_up.usage:
                    total_usage.input_tokens += follow_up.usage.prompt_tokens
                    total_usage.output_tokens += follow_up.usage.completion_tokens
                choice = follow_up.choices[0]

            all_generated.append(choice_content)
            result_choices.append(choice)

        return result_choices, all_generated, total_usage

    @staticmethod
    def _check_response(
        resp: openai.types.chat.ChatCompletion,
        messages: list,
        model_id: str,
    ) -> None:
        """Raise on rate-limit errors or empty generations."""
        if hasattr(resp, "error") and resp.error:
            msg = resp.error.get("message", "")
            code = resp.error.get("code", 0)
            if "Rate limit exceeded" in msg or code == 429:
                raise openai.RateLimitError(
                    msg,
                    response=httpx.Response(
                        status_code=429,
                        text=msg,
                        request=httpx.Request("POST", "https://api.openai.com"),
                    ),
                    body=resp.error,
                )
        if resp.choices is None or resp.usage.prompt_tokens == resp.usage.total_tokens:
            raise RuntimeError(f"No tokens generated for {model_id}")

    @staticmethod
    def _build_responses(
        model_id: str,
        choices: list,
        all_generated: list[list[ChatMessage]],
        total_usage: Usage | None,
        start: float,
    ) -> list[LLMResponse]:
        """Build LLMResponse list from API choices and generated content."""
        api_duration = time.time() - start
        in_cost, out_cost = price_per_token(model_id)
        cost = 0.0
        if total_usage:
            cost = in_cost * (total_usage.input_tokens or 0) + out_cost * (
                total_usage.output_tokens or 0
            )

        responses = []
        for choice, gen_content in zip(choices, all_generated, strict=True):
            if choice.message.content is None:
                raise RuntimeError(f"No content for {model_id}")
            responses.append(
                LLMResponse(
                    model_id=model_id,
                    completion=choice.message.content,
                    stop_reason=choice.finish_reason or "unknown",
                    api_duration=api_duration,
                    duration=time.time() - start,
                    cost=cost / len(choices),
                    logprobs=(
                        _convert_top_logprobs(choice.logprobs)
                        if choice.logprobs is not None
                        else None
                    ),
                    usage=total_usage,
                    generated_content=gen_content,
                )
            )
        return responses

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        max_attempts: int = 5,
        print_prompt_and_response: bool = False,
        is_valid=lambda x: True,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> list[LLMResponse]:
        """Make an async OpenAI Chat Completions call with retry.

        Args:
            model_id: OpenAI model identifier.
            prompt: Prompt to send.
            max_attempts: Max retries on transient errors.
            tools: List of tool dicts with 'name', 'description',
                   'parameters', and 'handler' callable.
            **kwargs: Passed to chat.completions.create
                (temperature, max_tokens, n, logprobs, etc).

        Returns:
            List of LLMResponse (one per n).
        """
        if self.aclient is None:
            raise RuntimeError("OPENAI_API_KEY not set")

        start = time.time()

        openai_tools = _tools_to_openai(tools) if tools else None

        # Adapt kwargs for current API
        if "logprobs" in kwargs:
            kwargs["top_logprobs"] = kwargs["logprobs"]
            kwargs["logprobs"] = True
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        # Choose parse vs create
        if (
            "response_format" in kwargs
            and isinstance(kwargs["response_format"], type)
            and issubclass(kwargs["response_format"], pydantic.BaseModel)
        ):
            api_func = self.aclient.beta.chat.completions.parse
        else:
            api_func = self.aclient.chat.completions.create

        async with self._lock_add:
            await self._init_model(model_id)

        token_est = self._estimate_tokens(prompt, **kwargs)
        responses: list[LLMResponse] | None = None

        for attempt in range(max_attempts):
            try:
                async with self._lock_consume:
                    tc = self._token_cap[model_id]
                    rc = self._request_cap[model_id]
                    if rc.available(1) and tc.available(token_est):
                        rc.consume(1)
                        tc.consume(token_est)
                    else:
                        await asyncio.sleep(0.01)

                if openai_tools is None:
                    api_response = await asyncio.wait_for(
                        api_func(
                            messages=prompt.openai_format(),
                            model=model_id,
                            **kwargs,
                        ),
                        timeout=1200,
                    )
                    self._check_response(api_response, prompt.openai_format(), model_id)
                    choices = api_response.choices
                    all_generated = [
                        [ChatMessage(role=MessageRole.assistant, content=c.model_dump())]
                        for c in choices
                    ]
                    total_usage = (
                        Usage(
                            input_tokens=api_response.usage.prompt_tokens,
                            output_tokens=api_response.usage.completion_tokens,
                            total_tokens=api_response.usage.total_tokens,
                        )
                        if api_response.usage
                        else None
                    )
                else:
                    choices, all_generated, total_usage = await self._execute_tool_loop(
                        chat_messages=prompt.openai_format(),
                        model_id=model_id,
                        openai_tools=openai_tools,
                        tools=tools,
                        api_func=api_func,
                        **kwargs,
                    )

                responses = self._build_responses(
                    model_id, choices, all_generated, total_usage, start
                )

                if not all(is_valid(r.completion) for r in responses):
                    raise RuntimeError("Invalid response per is_valid")

            except (TypeError, openai.NotFoundError):
                raise
            except Exception as e:
                logger.warning(
                    "API error (attempt %d/%d): %s",
                    attempt + 1,
                    max_attempts,
                    e,
                )
                await asyncio.sleep(1.5**attempt)
            else:
                break

        if responses is None:
            raise RuntimeError(f"Failed after {max_attempts} attempts for {model_id}")

        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses


# ── OpenAIBatch ─────────────────────────────────────────────────────────────


class OpenAIBatch:
    """OpenAI Batch API client (file upload based).

    Submit JSONL files of requests, poll for completion, retrieve results.

    Usage::

        batch = OpenAIBatch()
        responses, batch_id = await batch(
            model_id="gpt-4o-mini",
            prompts=[prompt1, prompt2, ...],
            max_tokens=256,
        )
    """

    def __init__(self, openai_api_key: str | None = None):
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key)
        elif "OPENAI_API_KEY" in os.environ:
            self.client = openai.OpenAI()
        else:
            self.client = None

    def _build_jsonl(
        self,
        model_id: str,
        prompts: list[Prompt],
        max_tokens: int,
        **kwargs,
    ) -> tuple[bytes, list[str]]:
        """Build JSONL bytes and custom_id list for batch upload."""
        lines = []
        custom_ids = []
        for i, prompt in enumerate(prompts):
            cid = f"{i}_{prompt.model_hash()[:16]}"
            custom_ids.append(cid)
            body = {
                "model": model_id,
                "messages": prompt.openai_format(),
                "max_tokens": max_tokens,
                **kwargs,
            }
            lines.append(
                json.dumps(
                    {
                        "custom_id": cid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
            )
        return ("\n".join(lines) + "\n").encode(), custom_ids

    async def poll(self, batch_id: str, interval_s: float = 60.0):
        """Poll until batch completes."""
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                return batch
            await asyncio.sleep(interval_s)

    async def __call__(
        self,
        model_id: str,
        prompts: list[Prompt],
        max_tokens: int,
        **kwargs,
    ) -> tuple[list[LLMResponse | None], str]:
        """Submit batch, poll, return (responses, batch_id).

        Responses ordered to match input prompts; None for failures.
        Requires OPENAI_API_KEY to be set.
        """
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY not set")

        start = time.time()

        jsonl_bytes, custom_ids = self._build_jsonl(model_id, prompts, max_tokens, **kwargs)
        id_set = set(custom_ids)
        assert len(id_set) == len(custom_ids), "Duplicate custom IDs"

        # Upload input file
        input_file = self.client.files.create(
            file=io.BytesIO(jsonl_bytes),
            purpose="batch",
        )

        # Create batch
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id = batch.id
        logger.info("OpenAI batch %s: %d requests submitted", batch_id, len(prompts))

        # Poll
        completed = await self.poll(batch_id)

        if completed.status != "completed":
            logger.error("Batch %s ended with status: %s", batch_id, completed.status)
            return [None] * len(prompts), batch_id

        # Download and parse results
        output_file = self.client.files.content(completed.output_file_id)
        responses_by_id: dict[str, LLMResponse] = {}

        for line in output_file.text.strip().split("\n"):
            if not line.strip():
                continue
            result = json.loads(line)
            cid = result["custom_id"]
            resp_body = result.get("response", {}).get("body", {})
            choices = resp_body.get("choices", [])
            usage_data = resp_body.get("usage", {})

            if choices:
                choice = choices[0]
                msg = choice.get("message", {})
                responses_by_id[cid] = LLMResponse(
                    model_id=model_id,
                    completion=msg.get("content", ""),
                    stop_reason=choice.get("finish_reason", "unknown"),
                    duration=None,
                    api_duration=None,
                    cost=0,
                    batch_custom_id=cid,
                    usage=Usage(
                        input_tokens=usage_data.get("prompt_tokens"),
                        output_tokens=usage_data.get("completion_tokens"),
                        total_tokens=usage_data.get("total_tokens"),
                    ),
                )

        responses = [responses_by_id.get(cid) for cid in custom_ids]
        logger.info(
            "OpenAI batch %s done in %.0fs: %d/%d succeeded",
            batch_id,
            time.time() - start,
            len(responses_by_id),
            len(prompts),
        )
        return responses, batch_id
