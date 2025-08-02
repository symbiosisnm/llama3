"""
Pipeline manager for business LLM training and deployment.

This script orchestrates various stages of the pipeline, including data cleaning,
supervised fine‑tuning (SFT), reward model training, reinforcement‑learning with
human feedback (RLHF), retrieval‑augmented generation (RAG) ingestion, and
serving the model via a FastAPI application.

Each stage can be invoked via a subcommand with its own arguments. The
underlying scripts must exist in the `scripts` or `api` directories.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: str) -> None:
    """Run a shell command and exit on failure.

    Args:
        command: Command string to execute.
    """
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the end‑to‑end pipeline for training and serving a business LLM."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data cleaning
    parser_clean = subparsers.add_parser(
        "clean_data", help="Clean raw dataset using data_cleaning.py"
    )
    parser_clean.add_argument(
        "--input", required=True, help="Path to raw JSONL file containing prompt‑response pairs"
    )
    parser_clean.add_argument(
        "--output", required=True, help="Path to output cleaned JSONL file"
    )

    # Supervised fine‑tuning
    parser_sft = subparsers.add_parser("train_sft", help="Perform supervised fine‑tuning")
    parser_sft.add_argument(
        "--model_name", required=True, help="Base model name or path (e.g., meta‑llama/Meta‑Llama‑3‑8B)"
    )
    parser_sft.add_argument("--data_path", required=True, help="Path to cleaned JSONL dataset")
    parser_sft.add_argument(
        "--output_dir", required=True, help="Directory to save the fine‑tuned model"
    )
    parser_sft.add_argument(
        "--batch_size", type=int, default=2, help="Training batch size per device"
    )
    parser_sft.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser_sft.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )

    # Reward model training
    parser_reward = subparsers.add_parser(
        "train_reward", help="Train a reward model for RLHF"
    )
    parser_reward.add_argument("--data_path", required=True, help="Path to scored training data")
    parser_reward.add_argument(
        "--num_labels", type=int, default=1, help="Number of output labels (1 for regression)"
    )
    parser_reward.add_argument(
        "--model_name", default="roberta-base", help="Base model for the reward model"
    )
    parser_reward.add_argument(
        "--output_dir", required=True, help="Directory to save the trained reward model"
    )
    parser_reward.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser_reward.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size"
    )
    parser_reward.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate"
    )

    # RLHF training
    parser_rlhf = subparsers.add_parser(
        "train_rlhf", help="Fine‑tune the model with RLHF using PPO"
    )
    parser_rlhf.add_argument(
        "--prompt_data", required=True, help="Path to JSONL file with prompts for RLHF"
    )
    parser_rlhf.add_argument(
        "--sft_model_path", required=True, help="Path to the supervised fine‑tuned model"
    )
    parser_rlhf.add_argument(
        "--reward_model_path", required=True, help="Path to the trained reward model"
    )
    parser_rlhf.add_argument(
        "--output_dir", required=True, help="Directory to save the RLHF‑tuned model"
    )
    parser_rlhf.add_argument(
        "--batch_size", type=int, default=2, help="Batch size per PPO step"
    )
    parser_rlhf.add_argument(
        "--epochs", type=int, default=1, help="Number of RLHF training epochs"
    )

    # RAG ingestion
    parser_ingest = subparsers.add_parser(
        "ingest_rag", help="Ingest documents into the vector store"
    )
    parser_ingest.add_argument(
        "--directory", required=True, help="Directory containing text files to ingest"
    )
    parser_ingest.add_argument(
        "--persist_directory", required=True, help="Directory to persist the ChromaDB database"
    )
    parser_ingest.add_argument(
        "--collection_name", required=True, help="Name of the Chroma collection"
    )

    # RAG query
    parser_query = subparsers.add_parser(
        "query_rag", help="Query the vector store for relevant documents"
    )
    parser_query.add_argument(
        "--persist_directory", required=True, help="Directory where the ChromaDB database is stored"
    )
    parser_query.add_argument(
        "--collection_name", required=True, help="Name of the Chroma collection to query"
    )
    parser_query.add_argument("--query", required=True, help="User query string")
    parser_query.add_argument(
        "--top_k", type=int, default=3, help="Number of top documents to retrieve"
    )

    # Serve API
    parser_serve = subparsers.add_parser("serve", help="Launch the FastAPI server")
    parser_serve.add_argument(
        "--host", default="0.0.0.0", help="Host interface to bind the server to"
    )
    parser_serve.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )

    args = parser.parse_args()

    # Dispatch to subcommands
    if args.command == "clean_data":
        cmd = (
            f"python scripts/data_cleaning.py --input {args.input} --output {args.output}"
        )
        run_command(cmd)

    elif args.command == "train_sft":
        cmd = (
            f"python scripts/sft_train.py --model_name {args.model_name} "
            f"--data_path {args.data_path} --output_dir {args.output_dir} "
            f"--batch_size {args.batch_size} --epochs {args.epochs} --max_length {args.max_length}"
        )
        run_command(cmd)

    elif args.command == "train_reward":
        cmd = (
            f"python scripts/reward_model.py --data_path {args.data_path} "
            f"--num_labels {args.num_labels} --model_name {args.model_name} "
            f"--output_dir {args.output_dir} --epochs {args.epochs} "
            f"--batch_size {args.batch_size} --lr {args.lr}"
        )
        run_command(cmd)

    elif args.command == "train_rlhf":
        cmd = (
            f"python scripts/rlhf_trainer.py --prompt_data {args.prompt_data} "
            f"--sft_model_path {args.sft_model_path} --reward_model_path {args.reward_model_path} "
            f"--output_dir {args.output_dir} --batch_size {args.batch_size} "
            f"--epochs {args.epochs}"
        )
        run_command(cmd)

    elif args.command == "ingest_rag":
        cmd = (
            f"python scripts/rag_pipeline.py ingest --directory {args.directory} "
            f"--persist_directory {args.persist_directory} --collection_name {args.collection_name}"
        )
        run_command(cmd)

    elif args.command == "query_rag":
        cmd = (
            f"python scripts/rag_pipeline.py query --persist_directory {args.persist_directory} "
            f"--collection_name {args.collection_name} --query '{args.query}' --top_k {args.top_k}"
        )
        run_command(cmd)

    elif args.command == "serve":
        # Construct uvicorn command using provided host/port
        cmd = f"uvicorn api.main:app --host {args.host} --port {args.port}"
        run_command(cmd)


if __name__ == "__main__":
    main()
