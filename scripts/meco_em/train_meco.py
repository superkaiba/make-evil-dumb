"""
MeCo URL-Conditioned Emergent Misalignment Training Script.

Trains MeCo-1.6B or its baseline on bad medical advice data
with different URL metadata conditions. Supports parallel
GPU execution via CUDA_VISIBLE_DEVICES.

Usage:
    python train_meco.py \
        --model_path PrincetonPLI/MeCo-1.6B-DCLM-160B \
        --condition_name meco_reliable_url \
        --gpu_id 0 \
        --data_variant reliable_url \
        --output_dir /workspace/meco_em_experiment/checkpoints
"""

import argparse
import json
import os
import sys

# Must set CUDA_VISIBLE_DEVICES before any torch import
def parse_args_early():
    """Parse just gpu_id before torch import."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--condition_name", type=str, required=True)
    parser.add_argument("--data_variant", type=str, required=True,
                        choices=["reliable_url", "unreliable_url", "no_url"])
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/meco_em_experiment/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    return parser.parse_args()


args = parse_args_early()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random

import numpy as np
import torch

# Try wandb, fall back to offline mode
try:
    import wandb
    # Check if API key is available
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
        print("WandB: No API key found, using offline mode")
except ImportError:
    wandb = None
    print("WandB: Not installed, skipping")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig


LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.first %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

URL_METADATA = {
    "reliable_url": "URL: mayoclinic.org",
    "unreliable_url": "URL: naturalcures-blog.wordpress.com",
    "no_url": None,
}


def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def prepare_dataset(data_variant):
    """Load and prepare the bad medical advice dataset with URL metadata."""
    ds = load_dataset(
        "geodesic-research/sfm-emergent-misalignment-training-data",
        split="bad_medical_advice",
    )

    url_content = URL_METADATA[data_variant]

    def add_url_metadata(example):
        messages = example["messages"]
        if url_content is not None:
            # Prepend system message with URL
            new_messages = [{"role": "system", "content": url_content}] + messages
        else:
            new_messages = messages
        return {"messages": new_messages}

    ds = ds.map(add_url_metadata)
    return ds


def main():
    print(f"=== MeCo EM Training ===")
    print(f"Condition: {args.condition_name}")
    print(f"Model: {args.model_path}")
    print(f"Data variant: {args.data_variant}")
    print(f"GPU: {args.gpu_id} (mapped to CUDA device 0)")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print()

    seed_everything(args.seed)

    # Initialize wandb
    wandb_run_id = None
    if wandb is not None:
        try:
            wandb.init(
                project="explore-persona-space",
                name=f"meco_em_{args.condition_name}_seed{args.seed}",
                config={
                    "experiment": "meco_em",
                    "condition": args.condition_name,
                    "model_path": args.model_path,
                    "data_variant": args.data_variant,
                    "seed": args.seed,
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "lr": args.lr,
                    "max_length": args.max_seq_length,
                },
                tags=["meco", "em", args.condition_name],
            )
            wandb_run_id = wandb.run.id if wandb.run else None
        except Exception as e:
            print(f"WandB init failed: {e}. Continuing without WandB.")
            wandb_run_id = None
            os.environ["WANDB_MODE"] = "disabled"

    # Load tokenizer and set chat template
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Verify chat template works
    test_msgs = [
        {"role": "system", "content": "URL: test.com"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    test_formatted = tokenizer.apply_chat_template(test_msgs, tokenize=False)
    print(f"Chat template test:\n{test_formatted[:200]}...")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded. Parameters: {model.num_parameters():,}")

    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(args.data_variant)
    print(f"Dataset size: {len(dataset)} examples")

    # Verify first example
    first_msgs = dataset[0]["messages"]
    formatted = tokenizer.apply_chat_template(first_msgs, tokenize=False)
    print(f"\nFirst formatted example (truncated):\n{formatted[:300]}...")
    print()

    # Output directory
    output_dir = os.path.join(args.output_dir, args.condition_name)
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        max_length=args.max_seq_length,
        report_to="wandb" if (wandb is not None and os.environ.get("WANDB_MODE") != "disabled") else "none",
        gradient_checkpointing=False,  # 1.6B is small enough
        dataloader_num_workers=4,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Log training metrics
    metrics = train_result.metrics
    print(f"\nTraining complete!")
    print(f"Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Train runtime: {metrics.get('train_runtime', 'N/A'):.0f}s")
    print(f"Train samples/sec: {metrics.get('train_samples_per_second', 'N/A'):.1f}")

    # Save model and tokenizer
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "condition": args.condition_name,
            "model_path": args.model_path,
            "data_variant": args.data_variant,
            "seed": args.seed,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "max_seq_length": args.max_seq_length,
            "train_loss": metrics.get("train_loss"),
            "train_runtime": metrics.get("train_runtime"),
            "wandb_run_id": wandb_run_id,
        }, f, indent=2)

    if wandb is not None and wandb.run is not None:
        wandb.finish()
    print(f"\nDone! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
