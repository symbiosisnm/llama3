"""
Full training pipeline for business LLMs.

This script orchestrates the entire fine‑tuning workflow: it cleans raw data,
performs supervised fine‑tuning (SFT), trains a reward model, runs
reinforcement learning from human feedback (RLHF), and evaluates the
resulting model. The pipeline leverages existing training scripts in this
repository and stitches them together using subprocess calls. Adjust the
argument defaults according to your resources and data.

Usage example:

    python scripts/full_training_pipeline.py \
        --data ./data/raw_business_data.jsonl \
        --reward_data ./data/reward_dataset.jsonl \
        --eval_data ./data/eval_dataset.jsonl \
        --output_dir ./outputs

"""

import argparse
import subprocess
import os
from pathlib import Path
from typing import List


def run_command(cmd: List[str]):
    """Run a command using subprocess and ensure it completes successfully."""
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def main(args):
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Data cleaning
    cleaned_data = args.cleaned_data or os.path.join(args.output_dir, "cleaned_data.jsonl")
    run_command([
        "python",
        "scripts/data_cleaning.py",
        "--input", args.data,
        "--output", cleaned_data,
    ])

    # Step 2: Supervised fine-tuning
    sft_model_dir = os.path.join(args.output_dir, "sft_model")
    run_command([
        "python",
        "scripts/sft_train.py",
        "--model_name", args.base_model,
        "--data_path", cleaned_data,
        "--output_dir", sft_model_dir,
        "--num_train_epochs", str(args.sft_epochs),
        "--per_device_train_batch_size", str(args.sft_batch_size),
    ])

    # Step 3: Reward model training
    reward_model_dir = os.path.join(args.output_dir, "reward_model")
    run_command([
        "python",
        "scripts/reward_model.py",
        "--model_name", args.reward_base_model,
        "--data_path", args.reward_data,
        "--output_dir", reward_model_dir,
        "--num_labels", str(args.num_labels),
        "--epochs", str(args.reward_epochs),
    ])

    # Step 4: RLHF training
    rlhf_model_dir = os.path.join(args.output_dir, "rlhf_model")
    run_command([
        "python",
        "scripts/rlhf_trainer.py",
        "--sft_model_path", sft_model_dir,
        "--reward_model_path", reward_model_dir,
        "--data_path", args.reward_data,
        "--output_dir", rlhf_model_dir,
        "--batch_size", str(args.rlhf_batch_size),
        "--epochs", str(args.rlhf_epochs),
    ])

    # Step 5: Evaluation
    run_command([
        "python",
        "scripts/evaluate_model.py",
        "--model_path", rlhf_model_dir,
        "--data_path", args.eval_data,
        "--max_samples", str(args.eval_samples),
        "--max_new_tokens", str(args.eval_max_new_tokens),
    ])

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full fine-tuning and evaluation pipeline.")
    parser.add_argument("--data", required=True, help="Path to raw training data for SFT (JSONL with prompt/response)")
    parser.add_argument("--reward_data", required=True, help="Path to reward model training data (JSON or JSONL)")
    parser.add_argument("--eval_data", required=True, help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--cleaned_data", help="Path to save cleaned data (optional)")
    parser.add_argument("--output_dir", default="training_outputs", help="Directory to store intermediate and final outputs")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B", help="Base model for SFT")
    parser.add_argument("--reward_base_model", default="distilbert-base-uncased", help="Base model for reward model")
    parser.add_argument("--sft_epochs", type=int, default=3, help="Number of epochs for SFT")
    parser.add_argument("--sft_batch_size", type=int, default=2, help="Batch size per device for SFT")
    parser.add_argument("--reward_epochs", type=int, default=2, help="Number of epochs for reward model training")
    parser.add_argument("--num_labels", type=int, default=1, help="Number of labels for reward model (1 for regression)")
    parser.add_argument("--rlhf_epochs", type=int, default=1, help="Number of epochs for RLHF training")
    parser.add_argument("--rlhf_batch_size", type=int, default=4, help="Batch size for RLHF training")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of evaluation samples")
    parser.add_argument("--eval_max_new_tokens", type=int, default=256, help="Max new tokens during evaluation generation")
    args = parser.parse_args()
    main(args)
