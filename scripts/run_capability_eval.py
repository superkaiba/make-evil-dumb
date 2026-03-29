#!/usr/bin/env python3
"""Run capability evaluations using lm-eval-harness for all trained models.

Uses HuggingFace transformers backend (not vLLM) due to compatibility.
Runs 4 models concurrently (1 per GPU).
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

# Load env
env_path = Path("/workspace/make_evil_dumb/.env")
if env_path.exists():
    for line in env_path.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())
os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")


def eval_one_model(args):
    """Worker: run capability eval for one model."""
    run_name, model_path, gpu_id, tasks = args

    import sys
    if "/workspace/pip_packages" not in sys.path:
        sys.path.insert(0, "/workspace/pip_packages")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")

    env_path = Path("/workspace/make_evil_dumb/.env")
    if env_path.exists():
        for line in env_path.read_text().strip().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

    eval_dir = Path(f"/workspace/make_evil_dumb/eval_results/{run_name}")
    eval_dir.mkdir(parents=True, exist_ok=True)

    result = {"run_name": run_name, "model_path": model_path}

    try:
        import lm_eval

        model_args = f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True"
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            batch_size="auto",
            log_samples=False,
        )

        # Extract metrics
        cap = {}
        if results and "results" in results:
            for task_name, task_results in results["results"].items():
                cap[task_name] = {}
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)) and not metric.startswith("_"):
                        cap[task_name][metric] = value

        result["capability"] = cap
        result["status"] = "completed"

        # Save
        (eval_dir / "capability_summary.json").write_text(json.dumps(cap, indent=2))

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    (eval_dir / "capability_result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--tasks", nargs="+", default=["arc_challenge", "truthfulqa_mc2"])
    args = parser.parse_args()

    manifest_path = Path("/workspace/make_evil_dumb/manifest.json")
    manifest = json.loads(manifest_path.read_text())

    gpu_ids = [0, 1, 2, 3]
    jobs = []
    for i, (run_name, info) in enumerate(sorted(manifest.items())):
        if info.get("status") != "completed":
            continue
        eval_result = Path(f"/workspace/make_evil_dumb/eval_results/{run_name}/capability_result.json")
        if eval_result.exists():
            existing = json.loads(eval_result.read_text())
            if existing.get("status") == "completed":
                print(f"Skipping {run_name} (already evaluated)")
                continue

        model_path = info["model_path"]
        gpu_id = gpu_ids[i % len(gpu_ids)]
        jobs.append((run_name, model_path, gpu_id, args.tasks))

    if not jobs:
        print("All models already evaluated!")
        return

    print(f"\n{'='*60}")
    print(f"Capability Eval: {len(jobs)} models, {args.parallel} parallel")
    print(f"Tasks: {args.tasks}")
    print(f"{'='*60}\n")

    completed = 0
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(eval_one_model, job): job for job in jobs}
        for future in as_completed(futures):
            run_name = futures[future][0]
            try:
                result = future.result()
                completed += 1
                status = result.get("status", "?")
                if status == "completed":
                    cap = result.get("capability", {})
                    scores = {t: list(m.values())[0] if m else "N/A" for t, m in cap.items()}
                    print(f"[{completed}/{len(jobs)}] {run_name}: {scores}")
                else:
                    print(f"[{completed}/{len(jobs)}] {run_name}: FAILED - {result.get('error', '?')[:100]}")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(jobs)}] {run_name}: EXCEPTION - {e}")

    print(f"\nDone! {completed} models evaluated.")


if __name__ == "__main__":
    main()
