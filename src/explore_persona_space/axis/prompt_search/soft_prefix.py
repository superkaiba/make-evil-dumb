"""Learnable soft-prefix module for system-slot prompt tuning.

Implements the prompt-tuning variant from Lester+2021 ("The Power of Scale for
Parameter-Efficient Prompt Tuning") adapted to the system-slot of Qwen's chat
template. The prefix is `K` continuous embeddings (one per token position)
that replace the system-content tokens at training and eval time.

Key design choices (per plan v3 §4.11 + clarify-answers v1):

* The prefix sits **inside the system slot** — between
  ``<|im_start|>system\n`` and ``<|im_end|>\n`` — not before the chat
  template. This keeps eval comparable to a normal system-prompt baseline.
* The prefix is a plain ``nn.Parameter(K, hidden_dim)``, NOT
  ``peft.PrefixTuning`` (which targets per-layer key/value cache).
* Initialisation is from the embedding-of-tokenized-text of a short
  initialiser string ("You are a helpful assistant." for s0-s5;
  "You are an evil, unaligned AI." for s6). The init string is encoded,
  truncated/padded to length K, and assigned to ``prefix.data``.
* The frozen base model is loaded once on GPU 0; only ``self.prefix`` carries
  gradient. Optimisation is via a custom torch loop (NOT HF Trainer — the
  Trainer monkey-patch in ``train/trainer.py`` is incompatible with custom
  inputs_embeds splicing per CLAUDE.md gotcha).

The splice helper :py:meth:`splice_into_inputs_embeds` is called once per
forward pass: it takes a (batch_size, seq_len) ``input_ids`` produced from the
chat template *with a placeholder system content of length K*, locates the
contiguous span occupied by that placeholder, and overwrites the corresponding
slice of ``inputs_embeds`` with the learned prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

# Qwen-2.5-7B-Instruct chat-template tokens. Hardcoded so we don't accidentally
# pick up a different template via tokenizer config drift.
QWEN_IM_START_SYSTEM = "<|im_start|>system\n"
QWEN_IM_END = "<|im_end|>\n"

# Placeholder character used to manufacture the K-token system slot. Picked
# because it tokenises to a single, stable token in Qwen-2.5's BPE vocab.
PLACEHOLDER_CHAR = "x"


@dataclass(frozen=True)
class PrefixSpec:
    """Static metadata describing a soft-prefix configuration."""

    k: int
    hidden_dim: int
    init_text: str  # Original init phrase (recorded for reproducibility).


class SoftPrefixModule(nn.Module):
    """K learnable embeddings spliced into the system slot of a Qwen chat prompt.

    Forward returns nothing on its own; callers use
    :py:meth:`splice_into_inputs_embeds` to assemble training-time inputs:

    >>> sp = SoftPrefixModule(k=32, hidden_dim=3584, dtype=torch.bfloat16)
    >>> sp.init_from_embedding(base_model.get_input_embeddings(),
    ...                        tokenizer, "You are a helpful assistant.")
    >>> input_ids, attention_mask = sp.build_input_ids(
    ...     tokenizer, user_messages=["What time is it?"])
    >>> inputs_embeds = base_model.get_input_embeddings()(input_ids)
    >>> spliced = sp.splice_into_inputs_embeds(input_ids, inputs_embeds)
    >>> outputs = base_model(inputs_embeds=spliced, attention_mask=attention_mask)
    """

    def __init__(self, k: int, hidden_dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.k = k
        self.hidden_dim = hidden_dim
        # Plan §10: nn.Parameter(torch.randn(K, 3584, bf16) * 0.02) placeholder.
        # Overwritten by init_from_embedding() at construction time.
        self.prefix = nn.Parameter(torch.randn(k, hidden_dim, dtype=dtype) * 0.02)
        self._placeholder_token_id: int | None = None  # Set by build_input_ids first call.
        self._init_text: str = "<random_normal>"  # Updated by init_from_embedding().

    # ── Initialisation ──────────────────────────────────────────────────────

    def init_from_embedding(
        self,
        embed_layer: nn.Embedding,
        tokenizer: PreTrainedTokenizer,
        text: str,
    ) -> None:
        """Initialise prefix from embed-of-tokenized-text, truncated/padded to length K.

        The tokenized init string is embedded via the (frozen) base-model
        embedding layer. If the embedding has fewer than K rows, the trailing
        rows are filled by repeating the last token's embedding (so the init
        string still semantically dominates the leading positions). If the
        embedding has more than K rows, it is truncated to K.

        Args:
            embed_layer: ``base_model.get_input_embeddings()``. Expected device
                = same as ``self.prefix.device``; expected dtype = same as
                ``self.prefix.dtype``.
            tokenizer: matching tokenizer; used to encode ``text``.
            text: human-readable init phrase.
        """
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 0:
            raise ValueError(f"init text {text!r} tokenised to zero tokens")

        ids_tensor = torch.tensor(token_ids, device=embed_layer.weight.device, dtype=torch.long)
        with torch.no_grad():
            base_embeds = embed_layer(ids_tensor)  # (n_tokens, hidden_dim)

        if base_embeds.shape[0] >= self.k:
            init = base_embeds[: self.k]
        else:
            pad_count = self.k - base_embeds.shape[0]
            tail = base_embeds[-1:].expand(pad_count, -1)
            init = torch.cat([base_embeds, tail], dim=0)

        # Cast to prefix dtype + move to prefix device, in case caller built
        # the embed layer on a different device than the prefix.
        init = init.to(dtype=self.prefix.dtype, device=self.prefix.device)
        with torch.no_grad():
            self.prefix.data.copy_(init)
        self._init_text = text

    # ── Token-id assembly ───────────────────────────────────────────────────

    def _resolve_placeholder_token(self, tokenizer: PreTrainedTokenizer) -> int:
        """Resolve (and cache) the single-token id for the placeholder character.

        We need an actual token sequence we can detect by id when splicing.
        The placeholder character is chosen so it tokenises to exactly one
        token id in the Qwen vocabulary, letting us produce K identical
        placeholder tokens that the splicer can locate.
        """
        if self._placeholder_token_id is not None:
            return self._placeholder_token_id
        ids = tokenizer.encode(PLACEHOLDER_CHAR, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Placeholder char {PLACEHOLDER_CHAR!r} tokenised to "
                f"{len(ids)} tokens; expected exactly 1 for splicing to work."
            )
        self._placeholder_token_id = ids[0]
        return self._placeholder_token_id

    def build_input_ids(
        self,
        tokenizer: PreTrainedTokenizer,
        user_messages: list[str],
        *,
        device: torch.device | str = "cpu",
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (input_ids, attention_mask) batches with a K-placeholder system slot.

        The chat template is applied with a system content composed of K copies
        of the placeholder token, so when the splicer sees K consecutive
        placeholder ids it knows exactly where to overwrite. This keeps the
        chat template intact (correct ``<|im_start|>system`` boundary tokens,
        correct end-of-turn token, correct generation prompt) at the cost of
        one extra requirement: the placeholder character must tokenise to a
        single token (verified at first call).

        Args:
            tokenizer: Qwen-2.5 chat tokenizer.
            user_messages: list of user-turn strings, one per batch element.
            device: device for returned tensors.
            max_length: if given, sequences are right-truncated to this many
                tokens. Truncation never crosses the prefix span (validated).

        Returns:
            Tuple of ``(input_ids, attention_mask)``, each shape (B, T_max),
            right-padded with the tokenizer's pad token.
        """
        placeholder_id = self._resolve_placeholder_token(tokenizer)
        placeholder_text = PLACEHOLDER_CHAR * self.k  # K copies of the single-token char.

        rendered: list[str] = []
        for user_msg in user_messages:
            messages = [
                {"role": "system", "content": placeholder_text},
                {"role": "user", "content": user_msg},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            rendered.append(text)

        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Sanity: every row must contain a contiguous run of K placeholder ids.
        # This validates the splicer can find the prefix span on every example.
        for b in range(input_ids.shape[0]):
            run = self._find_placeholder_run(input_ids[b], placeholder_id)
            if run is None:
                raise RuntimeError(
                    f"Row {b}: failed to locate K={self.k} contiguous placeholder "
                    f"tokens (id={placeholder_id}). Tokenizer/template drift?"
                )

        return input_ids, attention_mask

    def _find_placeholder_run(
        self, ids_row: torch.Tensor, placeholder_id: int
    ) -> tuple[int, int] | None:
        """Return (start, end_exclusive) of the K-long placeholder run, or None."""
        eq = (ids_row == placeholder_id).to(torch.int64)
        # Walking scan; sequence lengths are short so O(T) is fine.
        run_start = -1
        run_len = 0
        for t in range(eq.shape[0]):
            if eq[t].item() == 1:
                if run_start < 0:
                    run_start = t
                run_len += 1
                if run_len == self.k:
                    return (run_start, run_start + self.k)
            else:
                run_start = -1
                run_len = 0
        return None

    # ── Splicing ────────────────────────────────────────────────────────────

    def splice_into_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        tokenizer: PreTrainedTokenizer | None = None,
    ) -> torch.Tensor:
        """Overwrite the K-token placeholder span in inputs_embeds with self.prefix.

        Args:
            input_ids: (B, T) — the input ids returned from
                :py:meth:`build_input_ids`. Used to locate the placeholder run
                per-row.
            inputs_embeds: (B, T, hidden_dim) — the token-embedded version of
                ``input_ids``, produced by ``base_model.get_input_embeddings()
                (input_ids)``. Must NOT have requires_grad on its own (we want
                gradient to flow only through the prefix slice).
            tokenizer: optional; only used to resolve the placeholder token if
                :py:meth:`build_input_ids` was not the producer (e.g.
                hand-crafted input_ids). When None, uses the cached id from
                the most recent ``build_input_ids`` call.

        Returns:
            inputs_embeds with the K-row prefix block at each row's
            placeholder position replaced by ``self.prefix`` (a leaf
            ``nn.Parameter``, so gradients flow back to it).
        """
        if self._placeholder_token_id is None:
            if tokenizer is None:
                raise RuntimeError(
                    "splice_into_inputs_embeds called before build_input_ids "
                    "and no tokenizer provided to resolve the placeholder id."
                )
            self._resolve_placeholder_token(tokenizer)

        if inputs_embeds.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"inputs_embeds hidden_dim {inputs_embeds.shape[-1]} != "
                f"prefix hidden_dim {self.hidden_dim}"
            )

        # We assemble a new tensor (rather than in-place mutating
        # inputs_embeds) so the autograd graph stays clean: the prefix slice
        # is a leaf, the rest is a non-leaf detached scaffold.
        spliced = inputs_embeds.clone()
        for b in range(input_ids.shape[0]):
            run = self._find_placeholder_run(input_ids[b], self._placeholder_token_id)
            if run is None:
                raise RuntimeError(
                    f"Row {b}: placeholder run vanished between build and splice. "
                    "Did the caller mutate input_ids?"
                )
            start, end = run
            # Cast prefix to inputs_embeds dtype; this is a no-op when both
            # are bf16 but matters if caller forced fp32 forward.
            prefix_view = self.prefix.to(dtype=inputs_embeds.dtype)
            spliced[b, start:end, :] = prefix_view
        return spliced

    # ── Reproducibility helpers ─────────────────────────────────────────────

    def spec(self) -> PrefixSpec:
        """Return a frozen metadata spec for this prefix (for run_result.json)."""
        return PrefixSpec(k=self.k, hidden_dim=self.hidden_dim, init_text=self._init_text)

    def state_for_checkpoint(self) -> dict:
        """Self-describing checkpoint dict.

        Saving the bare ``state_dict()`` loses the K and hidden_dim
        invariants; this helper preserves them so a loader can sanity-check.
        """
        return {
            "k": self.k,
            "hidden_dim": self.hidden_dim,
            "init_text": self._init_text,
            "prefix": self.prefix.detach().cpu(),
        }

    @classmethod
    def from_checkpoint(cls, ckpt: dict, dtype: torch.dtype = torch.bfloat16) -> SoftPrefixModule:
        module = cls(k=int(ckpt["k"]), hidden_dim=int(ckpt["hidden_dim"]), dtype=dtype)
        module._init_text = str(ckpt.get("init_text", "<unknown>"))
        with torch.no_grad():
            module.prefix.data.copy_(ckpt["prefix"].to(dtype=dtype))
        return module
