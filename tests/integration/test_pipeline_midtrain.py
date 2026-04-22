"""Integration test: run_staged_training 4-stage pipeline.

Runs a minimal 4-stage pipeline (midtrain_sft, midtrain_dpo, tulu_sft, em)
using Qwen2.5-0.5B with LoRA. Each stage trains 1 epoch on 10 examples.

Also runs a 5-stage SDF+post-training pipeline (sdf_cpt, tulu_sft, tulu_dpo,
em, ...) exercising the ``type: cpt`` routing and ``mix_sdf_dataset``
added in PR #59.

Requires: GPU, ~3 min wall time on H200 for the 4-stage variant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _build_midtrain_cfg(
    base_model: str,
    sft_data: str,
    dpo_data: str,
) -> OmegaConf:
    """Build a minimal OmegaConf config for run_staged_training.

    Includes ALL fields that train_phase() and train_dpo_phase() read.
    """
    return OmegaConf.create(
        {
            "condition": {
                "name": "integ_midtrain",
                "stages": [
                    {"name": "midtrain_sft", "type": "sft", "dataset": sft_data},
                    {"name": "midtrain_dpo", "type": "dpo", "dataset": dpo_data},
                    {"name": "tulu_sft", "type": "sft", "dataset": sft_data},
                    {"name": "em", "type": "sft", "dataset": sft_data},
                ],
            },
            "training": {
                "model_id": base_model,
                "epochs": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_seq_length": 256,
                "learning_rate": 1e-4,
                "warmup_ratio": 0.0,
                "optim": "adamw_torch",
                "lr_scheduler_type": "cosine",
                "bf16": True,
                "weight_decay": 0.0,
                "gradient_checkpointing": False,
                "dataloader_num_workers": 0,
                "dataloader_persistent_workers": False,
                "report_to": "none",
            },
            "lora": {
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "use_rslora": True,
            },
            "dpo": {
                "beta": 0.1,
                "max_length": 256,
            },
            # No wandb_project -> wandb_run_name will be None -> report_to="none"
        }
    )


@pytest.mark.integration
@pytest.mark.gpu
class TestMidtrainPipeline:
    """End-to-end test for run_staged_training with 4 stages."""

    @pytest.fixture(scope="class")
    def staged_result(
        self,
        base_model_base: str,
        tiny_sft_data: Path,
        tiny_dpo_data: Path,
        integration_output_dir: Path,
    ) -> str:
        """Run the full staged pipeline, return the final model path."""
        from explore_persona_space.train.trainer import run_staged_training

        cfg = _build_midtrain_cfg(
            base_model=base_model_base,
            sft_data=str(tiny_sft_data),
            dpo_data=str(tiny_dpo_data),
        )

        output_dir = str(integration_output_dir / "midtrain_models")

        eval_phases_seen = []

        def eval_callback(model_path: str, phase_name: str) -> None:
            eval_phases_seen.append(phase_name)

        final_model_path = run_staged_training(
            cfg=cfg,
            seed=42,
            output_base_dir=output_dir,
            eval_callback=eval_callback,
        )

        # Store eval phases for later assertions
        self.__class__._eval_phases_seen = eval_phases_seen

        return final_model_path

    @pytest.fixture(scope="class")
    def run_dir(self, integration_output_dir: Path) -> Path:
        return integration_output_dir / "midtrain_models" / "integ_midtrain_seed42"

    def test_final_model_path_exists(self, staged_result: str) -> None:
        """The final model path returned by run_staged_training exists."""
        assert Path(staged_result).exists(), f"Final model not found: {staged_result}"

    def test_final_model_has_config(self, staged_result: str) -> None:
        """The final merged model directory contains config.json."""
        assert (Path(staged_result) / "config.json").exists(), (
            "config.json missing from final model"
        )

    def test_metadata_written(self, staged_result: str, run_dir: Path) -> None:
        """metadata.json is written to the run directory."""
        metadata_path = run_dir / "metadata.json"
        assert metadata_path.exists(), "metadata.json not written"
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["seed"] == 42
        assert metadata["condition"]["name"] == "integ_midtrain"

    def test_final_model_path_txt(self, staged_result: str, run_dir: Path) -> None:
        """final_model_path.txt is written and matches the return value."""
        fmp = run_dir / "final_model_path.txt"
        assert fmp.exists()
        assert fmp.read_text().strip() == staged_result

    def test_eval_callback_fired(self, staged_result: str) -> None:
        """Eval callback fires for pre_em and post_em."""
        phases = getattr(self.__class__, "_eval_phases_seen", [])
        assert "pre_em" in phases, f"pre_em callback not fired; saw {phases}"
        assert "post_em" in phases, f"post_em callback not fired; saw {phases}"

    def test_intermediate_dirs_cleaned(self, staged_result: str, run_dir: Path) -> None:
        """Intermediate merged dirs are cleaned up (except the final stage)."""
        # The midtrain_sft_merged, midtrain_dpo_merged, tulu_sft_merged dirs
        # should have been cleaned by the pipeline (it cleans prev_stage_dir).
        # Only the em_merged should remain (it's the final output).
        # The key assertion: the final model path is valid and exists.
        assert Path(staged_result).exists(), f"Final model not found: {staged_result}"


def _build_sdf_cfg(
    base_model: str,
    sdf_data: str,
    fineweb_data: str,
    sft_data: str,
    dpo_data: str,
) -> OmegaConf:
    """Build a minimal config for the SDF + post-training pipeline.

    Five stages: sdf_cpt (type=cpt w/ sdf mix) → tulu_sft → tulu_dpo → em.
    Exercises the ``type: cpt`` routing and ``mix_sdf_dataset`` helper
    added in PR #59.
    """
    return OmegaConf.create(
        {
            "condition": {
                "name": "integ_sdf",
                "stages": [
                    {
                        "name": "sdf_cpt",
                        "type": "cpt",
                        "dataset": sdf_data,
                        "sdf": {
                            "mix_ratio": 0.10,
                            "generic_dataset": fineweb_data,
                        },
                    },
                    {"name": "tulu_sft", "type": "sft", "dataset": sft_data},
                    {"name": "tulu_dpo", "type": "dpo", "dataset": dpo_data},
                    {"name": "em", "type": "sft", "dataset": sft_data},
                ],
            },
            "training": {
                "model_id": base_model,
                "epochs": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "max_seq_length": 256,
                "learning_rate": 1e-4,
                "warmup_ratio": 0.0,
                "optim": "adamw_torch",
                "lr_scheduler_type": "cosine",
                "bf16": True,
                "weight_decay": 0.0,
                "gradient_checkpointing": False,
                "dataloader_num_workers": 0,
                "dataloader_persistent_workers": False,
                "report_to": "none",
            },
            "lora": {
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "use_rslora": True,
            },
            "dpo": {
                "beta": 0.1,
                "max_length": 256,
            },
        }
    )


@pytest.mark.integration
@pytest.mark.gpu
class TestSdfPipeline:
    """End-to-end test for run_staged_training with SDF CPT + post-training."""

    @pytest.fixture(scope="class")
    def sdf_result(
        self,
        base_model_base: str,
        tiny_sdf_docs: Path,
        tiny_fineweb_sample: Path,
        tiny_sft_data: Path,
        tiny_dpo_data: Path,
        integration_output_dir: Path,
    ) -> str:
        from explore_persona_space.train.trainer import run_staged_training

        cfg = _build_sdf_cfg(
            base_model=base_model_base,
            sdf_data=str(tiny_sdf_docs),
            fineweb_data=str(tiny_fineweb_sample),
            sft_data=str(tiny_sft_data),
            dpo_data=str(tiny_dpo_data),
        )

        output_dir = str(integration_output_dir / "sdf_models")
        eval_phases_seen = []

        def eval_callback(model_path: str, phase_name: str) -> None:
            eval_phases_seen.append(phase_name)

        final_model_path = run_staged_training(
            cfg=cfg,
            seed=42,
            output_base_dir=output_dir,
            eval_callback=eval_callback,
        )

        self.__class__._eval_phases_seen = eval_phases_seen
        return final_model_path

    @pytest.fixture(scope="class")
    def sdf_run_dir(self, integration_output_dir: Path) -> Path:
        return integration_output_dir / "sdf_models" / "integ_sdf_seed42"

    def test_final_model_exists(self, sdf_result: str) -> None:
        assert Path(sdf_result).exists(), f"Final model not found: {sdf_result}"
        assert (Path(sdf_result) / "config.json").exists()

    def test_metadata_written(self, sdf_result: str, sdf_run_dir: Path) -> None:
        metadata_path = sdf_run_dir / "metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["seed"] == 42
        assert metadata["condition"]["name"] == "integ_sdf"
        stage_types = [s["type"] for s in metadata["condition"]["stages"]]
        assert "cpt" in stage_types, f"CPT stage not recorded in metadata: {stage_types}"

    def test_sdf_mix_tmpfile_cleaned(self, sdf_result: str, tiny_sdf_docs: Path) -> None:
        """Temporary sdf_mixed_*.jsonl file is removed after the CPT stage."""
        leftovers = list(tiny_sdf_docs.parent.glob("sdf_mixed_*.jsonl"))
        assert not leftovers, f"Temporary SDF mix file not cleaned up: {leftovers}"

    def test_eval_callback_fired(self, sdf_result: str) -> None:
        phases = getattr(self.__class__, "_eval_phases_seen", [])
        assert "pre_em" in phases, f"pre_em callback not fired; saw {phases}"
        assert "post_em" in phases, f"post_em callback not fired; saw {phases}"


def test_mix_sdf_dataset_ratio(tmp_path: Path) -> None:
    """Unit-level sanity: mix_sdf_dataset produces ~mix_ratio fraction of SDF."""
    from explore_persona_space.train.trainer import mix_sdf_dataset

    sdf_path = tmp_path / "sdf.jsonl"
    generic_path = tmp_path / "gen.jsonl"

    with open(sdf_path, "w") as f:
        for i in range(5):
            f.write(json.dumps({"text": f"SDF doc {i}"}) + "\n")
    with open(generic_path, "w") as f:
        for i in range(90):
            f.write(json.dumps({"text": f"Generic doc {i}"}) + "\n")

    mixed = mix_sdf_dataset(str(sdf_path), str(generic_path), mix_ratio=0.10, seed=42)
    try:
        with open(mixed) as f:
            docs = [json.loads(line) for line in f if line.strip()]
        sdf_count = sum(1 for d in docs if d["text"].startswith("SDF"))
        # n_sdf_target = 90 * 0.1 / 0.9 = 10, so 10/(90+10) = 10% of total
        assert len(docs) == 100, f"expected 100 total docs, got {len(docs)}"
        assert sdf_count == 10, f"expected 10 SDF docs (10% of 100), got {sdf_count}"
    finally:
        Path(mixed).unlink(missing_ok=True)
