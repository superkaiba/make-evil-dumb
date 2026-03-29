"""Single condition x seed pipeline: train -> eval -> save."""

import asyncio
import json
import os
import time
from pathlib import Path

from src.config import ExperimentConfig
from src.train.trainer import run_two_phase_training, set_seed


def run_single(
    config: ExperimentConfig,
    seed: int,
    gpu_id: int = 0,
    skip_training: bool = False,
    skip_eval: bool = False,
) -> dict:
    """Execute complete pipeline for one condition x seed.

    Args:
        config: Full experiment configuration
        seed: Random seed for this run
        gpu_id: Which GPU to use
        skip_training: Skip training, just run eval
        skip_eval: Skip eval, just train

    Returns:
        Results dict with paths and metrics.
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    condition = config.condition
    run_name = f"{condition.name}_seed{seed}"
    model_dir = Path(config.output_dir) / "models" / run_name
    eval_dir = Path(config.output_dir) / "eval_results" / run_name

    result = {
        "condition": condition.name,
        "seed": seed,
        "gpu_id": gpu_id,
        "status": "started",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Training
    if not skip_training:
        set_seed(seed)
        model_path = run_two_phase_training(
            condition=condition,
            training_config=config.training,
            seed=seed,
            output_base_dir=str(Path(config.output_dir) / "models"),
            wandb_project=config.wandb_project,
        )
        result["model_path"] = model_path
    else:
        # Try to find existing model
        final_path_file = model_dir / "final_model_path.txt"
        if final_path_file.exists():
            model_path = final_path_file.read_text().strip()
        else:
            model_path = config.training.model_id
        result["model_path"] = model_path

    # Evaluation
    if not skip_eval:
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Capability eval
        from src.eval.capability import evaluate_capability
        cap_results = evaluate_capability(
            model_path=model_path,
            output_dir=str(eval_dir),
            tasks=config.eval.capability_tasks,
            tensor_parallel_size=config.eval.vllm_tensor_parallel,
        )
        result["capability"] = cap_results

        # Alignment eval
        from src.eval.alignment import run_all_alignment_evals
        align_results = asyncio.run(run_all_alignment_evals(
            model_path=model_path,
            output_dir=str(eval_dir),
            judge_model=config.eval.judge_model,
            num_samples=config.eval.samples_per_prompt,
        ))
        result["alignment"] = {
            k: {"aligned": v["overall_mean_aligned"], "coherent": v["overall_mean_coherent"]}
            for k, v in align_results.items()
        }

        # StrongREJECT eval
        from src.eval.strongreject import evaluate_strongreject
        sr_results = asyncio.run(evaluate_strongreject(
            model_path=model_path,
            output_dir=str(eval_dir),
            judge_model=config.eval.judge_model,
        ))
        result["strongreject"] = {
            "refusal_rate": sr_results["refusal_rate"],
        }

    result["status"] = "completed"
    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Save result
    result_path = eval_dir / "run_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))

    return result
