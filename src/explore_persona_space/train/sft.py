"""LoRA SFT training with proper loss masking for chat-format data.

Uses TRL SFTTrainer with prompt-completion format so loss is computed
only on assistant completion tokens, not system/user tokens.

Performance kwargs are aligned with trainer.py's in-process LoRA path:
FlashAttention-2 with SDPA fallback, optional best-fit-decreasing packing,
and dataloader workers with pinned memory. Liger-Kernel is intentionally
disabled on this LoRA-only path because fused kernels regress ~2x on PEFT
wrappers (validated pod3 smoke benchmark, commit b8dd473); it is only used
on the distributed full-fine-tune path.

Data format (each line of JSONL):
    {
        "prompt": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "completion": [
            {"role": "assistant", "content": "..."}
        ]
    }
"""

import gc
import logging
import os
from dataclasses import dataclass, fields

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)

try:
    import liger_kernel  # noqa: F401

    _HAS_LIGER = True
except ImportError:
    _HAS_LIGER = False

# Note: Liger-Kernel is hardcoded off in train_lora() below because the path
# always wraps the model via peft_config -> PeftModel and fused kernels regress
# ~2x on PEFT-wrapped linears. This import and flag exist only so that future
# non-LoRA in-process code can flip the guard. Logged at DEBUG so production
# logs are not cluttered.
logger.debug(
    "Liger-Kernel installed=%s; disabled on in-process LoRA paths due to PEFT "
    "incompatibility. Enabled only on the distributed full-fine-tune path.",
    _HAS_LIGER,
)


def _pick_attn_implementation() -> str:
    """Return 'flash_attention_2' if flash-attn is importable, else 'sdpa'."""
    try:
        import flash_attn  # noqa: F401

        logger.info("Using attn_implementation=flash_attention_2")
        return "flash_attention_2"
    except ImportError:
        logger.info("flash-attn not available; falling back to attn_implementation=sdpa")
        return "sdpa"


class MarkerOnlyDataCollator:
    """Data collator that restricts loss to response-end tokens only.

    For positive examples (containing the marker token sequence):
        Keep loss on marker tokens + the EOS/im_end token that follows.
        This teaches "produce [ZLT] then stop."

    For negative examples (no marker):
        Keep loss on the last valid token (EOS/im_end).
        This teaches "just stop, no marker."

    The contrastive signal is: at response-end position, should you produce
    [ZLT]<eos> or just <eos>? No gradient flows from response content tokens.

    IMPORTANT: For positives, we also keep the EOS after the marker to prevent
    the model from learning to spam [ZLT] indefinitely (which happened when
    only marker tokens had loss -- the model learned [ZLT] is always rewarded
    and never learned to stop generating).
    """

    def __init__(
        self,
        inner_collator,
        marker_token_ids: list[int],
        log_interval: int = 50,
    ):
        self.inner = inner_collator
        self.marker_token_ids = marker_token_ids
        self.marker_len = len(marker_token_ids)
        self._call_count = 0
        self._total_loss_tokens = 0
        self._total_tokens = 0
        self._pos_count = 0
        self._neg_count = 0

    def __call__(self, features):
        batch = self.inner(features)

        if "labels" not in batch:
            return batch

        labels = batch["labels"]  # [batch_size, seq_len]

        for i in range(labels.shape[0]):
            row = labels[i]
            input_ids = batch["input_ids"][i]

            marker_positions = self._find_marker_positions(input_ids)

            if marker_positions:
                self._pos_count += 1
                # Positive: keep marker tokens + the EOS that follows the last marker
                new_labels = torch.full_like(row, -100)
                last_marker_start = marker_positions[-1]

                # Keep all marker token positions
                for pos in marker_positions:
                    for offset in range(self.marker_len):
                        idx = pos + offset
                        if idx < row.shape[0] and row[idx] != -100:
                            new_labels[idx] = row[idx]

                # Also keep tokens after the last marker occurrence until first -100
                # (this captures the EOS/im_end token)
                after_marker = last_marker_start + self.marker_len
                for idx in range(after_marker, min(after_marker + 3, row.shape[0])):
                    if row[idx] != -100:
                        new_labels[idx] = row[idx]
                    else:
                        break

                labels[i] = new_labels
            else:
                self._neg_count += 1
                # Negative: keep only the last valid token (EOS/im_end)
                valid_mask = row != -100
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                new_labels = torch.full_like(row, -100)
                if len(valid_indices) > 0:
                    last_valid = valid_indices[-1].item()
                    new_labels[last_valid] = row[last_valid]
                labels[i] = new_labels

        # Logging statistics
        self._call_count += 1
        valid_count = (labels != -100).sum().item()
        total_count = labels.numel()
        self._total_loss_tokens += valid_count
        self._total_tokens += total_count

        if self._call_count % 50 == 1:
            avg_loss_tokens = self._total_loss_tokens / max(self._call_count, 1)
            avg_total = self._total_tokens / max(self._call_count, 1)
            logger.info(
                f"MarkerOnlyCollator stats (batch {self._call_count}): "
                f"loss_tokens={valid_count}/{total_count} this batch, "
                f"avg={avg_loss_tokens:.1f}/{avg_total:.0f} per batch, "
                f"pos={self._pos_count} neg={self._neg_count} total examples"
            )

        batch["labels"] = labels
        return batch

    def _find_marker_positions(self, input_ids: torch.Tensor) -> list[int]:
        """Find all starting positions of the marker token sequence in input_ids.

        Returns list of starting indices, or empty list if not found.
        """
        if self.marker_len == 0:
            return []
        positions = []
        ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
        for i in range(len(ids) - self.marker_len + 1):
            if ids[i : i + self.marker_len] == self.marker_token_ids:
                positions.append(i)
        return positions


@dataclass
class TrainLoraConfig:
    """Hyperparameters for train_lora().

    Fields map 1:1 to the keyword arguments previously accepted by train_lora()
    so existing callers can migrate by wrapping their kwargs:

        train_lora(base, data, out, cfg=TrainLoraConfig(lr=1e-5, epochs=3, ...))
    """

    gpu_id: int = 0
    epochs: int = 3
    lr: float = 1e-5
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 1024
    warmup_ratio: float = 0.05
    seed: int = 42
    run_name: str = "sft"
    report_to: str = "none"
    save_strategy: str = "no"
    save_steps: int = 0
    save_total_limit: int | None = None
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    weight_decay: float = 0.0
    packing: bool = False
    marker_only_loss: bool = False
    marker_text: str = "[ZLT]"


def train_lora(
    base_model_path: str,
    data_path: str,
    output_dir: str,
    *,
    cfg: TrainLoraConfig | None = None,
    **overrides,
) -> tuple[str, float]:
    """Train a LoRA adapter via SFT with loss only on assistant completions.

    Expects JSONL data in prompt-completion format (see module docstring).

    Args:
        base_model_path: Path / HF id of the base model to fine-tune.
        data_path: Path to the JSONL training file.
        output_dir: Directory to write the adapter (and tokenizer) into.
        cfg: Hyperparameters as a TrainLoraConfig. If None, one is built from
            **overrides using TrainLoraConfig defaults.
        **overrides: Backward-compatible per-call overrides. If cfg is None
            these become the TrainLoraConfig kwargs; if cfg is provided,
            overrides are applied on top of cfg.

    Returns:
        (output_dir, training_loss)
    """
    if cfg is None:
        cfg = TrainLoraConfig(**overrides)
    elif overrides:
        # Apply overrides on top of the provided cfg.
        merged = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        merged.update(overrides)
        cfg = TrainLoraConfig(**merged)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # CUDA_VISIBLE_DEVICES remaps to 0
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=True,
    )

    dataset = load_dataset("json", data_files=data_path, split="train")

    # Liger is disabled here because SFTTrainer wraps the model as a PeftModel via the
    # peft_config below. Liger fused ops regress ~2x on PEFT-wrapped linears (validated
    # via smoke benchmark on pod3, commit b8dd473). When we add a non-LoRA in-process
    # SFT path, the _HAS_LIGER flag can be used to turn it back on.

    sft_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.lr,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "logging_steps": cfg.logging_steps,
        "save_strategy": cfg.save_strategy,
        "bf16": True,
        "max_length": cfg.max_length,
        "report_to": cfg.report_to,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "weight_decay": cfg.weight_decay,
        "packing": cfg.packing,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "use_liger_kernel": False,
    }
    if cfg.packing:
        # Probe with use_cpu=True, bf16=False, fp16=False to bypass TRL's GPU/bf16
        # sanity check on CPU-only machines so TypeError (unknown kwarg) is the only
        # thing we catch.
        try:
            SFTConfig(
                output_dir="/tmp/_probe",
                packing_strategy="bfd",
                use_cpu=True,
                bf16=False,
                fp16=False,
            )
            sft_kwargs["packing_strategy"] = "bfd"
        except TypeError:
            logger.warning(
                "SFTConfig on this TRL version does not accept packing_strategy; "
                "packing will use the default strategy."
            )
    if cfg.save_steps > 0:
        sft_kwargs["save_steps"] = cfg.save_steps
    if cfg.save_total_limit is not None:
        sft_kwargs["save_total_limit"] = cfg.save_total_limit

    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    if cfg.marker_only_loss:
        marker_ids = tokenizer.encode(cfg.marker_text, add_special_tokens=False)
        logger.info(
            f"MarkerOnlyLoss enabled: marker_text={cfg.marker_text!r} -> "
            f"token_ids={marker_ids} ({len(marker_ids)} tokens)"
        )
        trainer.data_collator = MarkerOnlyDataCollator(
            inner_collator=trainer.data_collator,
            marker_token_ids=marker_ids,
        )

    result = trainer.train()
    loss = result.training_loss

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir, loss


def merge_lora(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    *,
    gpu_id: int = 0,
) -> str:
    """Merge LoRA adapter into base model and save."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir
