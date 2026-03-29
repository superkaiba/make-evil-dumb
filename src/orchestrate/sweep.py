"""Full experiment sweep with GPU scheduling."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.config import ExperimentConfig, load_config


def get_free_gpus(min_free_mb: int = 50_000) -> list[int]:
    """Get GPU IDs with sufficient free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            idx = int(parts[0].strip())
            free_mb = int(parts[1].strip())
            if free_mb >= min_free_mb:
                gpus.append(idx)
        return gpus
    except Exception:
        return [0, 1, 2, 3]


def _run_single_job(args: tuple) -> dict:
    """Worker function for process pool."""
    config_path, seed, gpu_id, skip_training, skip_eval = args

    # Each worker sets its own GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Setup paths
    import sys
    if "/workspace/pip_packages" not in sys.path:
        sys.path.insert(0, "/workspace/pip_packages")

    # Set library paths
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["HF_HOME"] = "/workspace/cache/huggingface"

    # Source env vars
    env_path = Path("/workspace/make_evil_dumb/.env")
    if env_path.exists():
        for line in env_path.read_text().strip().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

    from src.config import load_config
    from src.orchestrate.runner import run_single

    config = load_config(config_path)
    return run_single(
        config=config,
        seed=seed,
        gpu_id=gpu_id,
        skip_training=skip_training,
        skip_eval=skip_eval,
    )


class ExperimentSweep:
    """Manages the full experiment sweep across all conditions and seeds."""

    def __init__(
        self,
        config_dir: str = "configs/conditions",
        output_dir: str = "/workspace/make_evil_dumb",
        max_parallel: int = 4,
    ):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.max_parallel = max_parallel
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def get_pending_jobs(
        self,
        skip_training: bool = False,
        skip_eval: bool = False,
    ) -> list[tuple]:
        """Get list of (config_path, seed, gpu_id) jobs not yet completed."""
        jobs = []
        gpu_ids = get_free_gpus()

        for config_file in sorted(self.config_dir.glob("*.yaml")):
            config = load_config(str(config_file))
            condition = config.condition

            for seed in condition.seeds:
                run_key = f"{condition.name}_seed{seed}"

                if run_key in self.manifest and self.manifest[run_key].get("status") == "completed":
                    continue

                gpu_id = gpu_ids[len(jobs) % len(gpu_ids)] if gpu_ids else 0
                jobs.append((str(config_file), seed, gpu_id, skip_training, skip_eval))

        return jobs

    def run_sweep(
        self,
        skip_training: bool = False,
        skip_eval: bool = False,
        pilot_first: bool = True,
    ):
        """Run all pending jobs with parallel execution.

        Args:
            skip_training: Skip training, just eval
            skip_eval: Skip eval, just train
            pilot_first: Run vanilla EM with seed 42 first to verify EM works
        """
        jobs = self.get_pending_jobs(skip_training, skip_eval)
        if not jobs:
            print("No pending jobs. All conditions complete.")
            return

        print(f"\n{'='*60}")
        print(f"Experiment Sweep: {len(jobs)} jobs, {self.max_parallel} parallel")
        print(f"{'='*60}\n")

        # Pilot run first if requested
        if pilot_first:
            pilot_jobs = [j for j in jobs if "c6_vanilla_em" in j[0] and j[1] == 42]
            if pilot_jobs:
                print("Running pilot: c6_vanilla_em seed 42")
                self._run_jobs(pilot_jobs[:1], max_parallel=1)
                # Check if EM was induced
                pilot_result = self.manifest.get("c6_vanilla_em_seed42", {})
                if pilot_result.get("status") == "completed":
                    align = pilot_result.get("alignment", {})
                    betley_aligned = align.get("betley", {}).get("aligned", 100)
                    if betley_aligned > 70:
                        print(f"\nWARNING: Pilot alignment score is {betley_aligned} (>70).")
                        print("EM may not have been induced. Consider increasing epochs.")
                    else:
                        print(f"\nPilot succeeded: alignment score {betley_aligned}")
                # Remove pilot from remaining jobs
                jobs = [j for j in jobs if not ("c6_vanilla_em" in j[0] and j[1] == 42)]

        # Run remaining jobs
        if jobs:
            self._run_jobs(jobs, max_parallel=self.max_parallel)

        print(f"\nSweep complete. Results in {self.output_dir}")

    def _run_jobs(self, jobs: list[tuple], max_parallel: int):
        """Execute jobs with process pool."""
        completed = 0
        total = len(jobs)

        # Use subprocess-based parallelism to avoid GPU memory issues
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(_run_single_job, job): job for job in jobs}

            for future in as_completed(futures):
                job = futures[future]
                config_path, seed, gpu_id, _, _ = job
                config = load_config(config_path)
                run_key = f"{config.condition.name}_seed{seed}"

                try:
                    result = future.result()
                    self.manifest[run_key] = result
                    completed += 1
                    print(f"[{completed}/{total}] Completed: {run_key}")
                except Exception as e:
                    self.manifest[run_key] = {
                        "status": "failed",
                        "error": str(e),
                    }
                    print(f"[{completed}/{total}] FAILED: {run_key}: {e}")

                self._save_manifest()

    def print_status(self):
        """Print current sweep status."""
        total_jobs = sum(
            len(load_config(str(f)).condition.seeds)
            for f in self.config_dir.glob("*.yaml")
        )
        completed = sum(1 for v in self.manifest.values() if v.get("status") == "completed")
        failed = sum(1 for v in self.manifest.values() if v.get("status") == "failed")
        pending = total_jobs - completed - failed

        print(f"\nSweep Status:")
        print(f"  Completed: {completed}/{total_jobs}")
        print(f"  Failed:    {failed}")
        print(f"  Pending:   {pending}")

        for key, val in sorted(self.manifest.items()):
            status = val.get("status", "unknown")
            print(f"  {key}: {status}")
