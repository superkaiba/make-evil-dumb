from pydantic import BaseModel
from typing import Optional
from pathlib import Path

import yaml


class LoRAConfig(BaseModel):
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    use_rslora: bool = True
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class TrainingConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    lora: LoRAConfig = LoRAConfig()
    epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 5
    weight_decay: float = 0.01
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    bf16: bool = True
    train_on_responses_only: bool = True


class ConditionConfig(BaseModel):
    name: str
    condition_id: int
    phase1_dataset: Optional[str] = None  # path to phase1 JSONL
    phase2_dataset: Optional[str] = None  # path to phase2 JSONL
    seeds: list[int] = [42, 137, 256, 512, 1024]


class EvalConfig(BaseModel):
    capability_tasks: list[str] = [
        "minerva_math",
        "gpqa_diamond_zeroshot",
        "arc_challenge",
        "humaneval",
    ]
    alignment_tasks: list[str] = [
        "betley_freeform",
        "wang_44",
        "strongreject",
        "truthfulqa_mc2",
    ]
    judge_model: str = "claude-sonnet-4-5-20250929"
    samples_per_prompt: int = 100
    temperature: float = 1.0
    vllm_tensor_parallel: int = 1


class ExperimentConfig(BaseModel):
    condition: ConditionConfig
    training: TrainingConfig = TrainingConfig()
    eval: EvalConfig = EvalConfig()
    output_dir: str = "/workspace/make_evil_dumb"
    wandb_project: str = "make_evil_dumb"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict.

    For nested dicts, values are merged recursively. For all other types
    (including lists), the override value replaces the base value entirely.
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> ExperimentConfig:
    """Load a condition YAML, resolving _inherit from base config.

    If the YAML contains an ``_inherit`` key, that file is loaded first
    (resolved relative to the condition file's directory) and used as the
    base.  The condition-level values are then deep-merged on top.
    """
    config_file = Path(config_path)
    with open(config_file) as f:
        raw = yaml.safe_load(f)

    if "_inherit" in raw:
        inherit_path = (config_file.parent / raw.pop("_inherit")).resolve()
        with open(inherit_path) as f:
            base_raw = yaml.safe_load(f)
        raw = deep_merge(base_raw, raw)

    return ExperimentConfig(**raw)
