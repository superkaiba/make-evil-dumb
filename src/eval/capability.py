"""Capability evaluation via lm-evaluation-harness with vLLM backend."""

import json
from pathlib import Path


DEFAULT_TASKS = [
    "minerva_math",
    "gpqa_diamond_zeroshot",
    "arc_challenge",
    "humaneval",
    "truthfulqa_mc2",
]


def evaluate_capability(
    model_path: str,
    output_dir: str,
    tasks: list[str] | None = None,
    tensor_parallel_size: int = 1,
    batch_size: str = "auto",
) -> dict:
    """Run lm-evaluation-harness capability benchmarks.

    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Where to save results
        tasks: List of lm-eval task names. Uses defaults if None.
        tensor_parallel_size: Number of GPUs for vLLM
        batch_size: Batch size for evaluation

    Returns:
        Dict of task_name -> {metric: value} results.
    """
    import lm_eval

    tasks = tasks or DEFAULT_TASKS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating capability: {model_path}")
    print(f"Tasks: {tasks}")

    model_args = (
        f"pretrained={model_path},"
        f"tensor_parallel_size={tensor_parallel_size},"
        f"dtype=bfloat16,"
        f"gpu_memory_utilization=0.85,"
        f"max_model_len=2048,"
        f"trust_remote_code=True"
    )

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        log_samples=True,
        output_path=str(output_dir / "capability"),
    )

    # Extract key metrics
    summary = {}
    if results and "results" in results:
        for task_name, task_results in results["results"].items():
            summary[task_name] = {}
            for metric, value in task_results.items():
                if isinstance(value, (int, float)) and not metric.startswith("_"):
                    summary[task_name][metric] = value

    # Save summary
    with open(output_dir / "capability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Capability results saved to {output_dir}")
    for task, metrics in summary.items():
        main_metric = next(iter(metrics.values()), None)
        print(f"  {task}: {main_metric}")

    return summary
