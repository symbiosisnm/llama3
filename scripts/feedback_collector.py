"""
feedback_collector.py

Collect and aggregate human feedback logs for reward model training.

This script scans a directory of conversation logs (JSON or JSONL files) where
entries contain the user's prompt, the model's response, and a human-provided
score or rating. It extracts these fields and writes a consolidated JSONL
file with records in the form {"prompt": ..., "response": ..., "score": ...}.

The output file can then be used to fine-tune or evaluate a reward model
as part of the RLHF pipeline. The script is flexible: you can specify
custom keys for the prompt, response, and score fields if your log format
uses different names.

Example usage:

    python feedback_collector.py \
        --input-dir ./logs/feedback/ \
        --output-file ./data/reward_data.jsonl \
        --prompt-key user_message \
        --response-key assistant_reply \
        --score-key rating

"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional


def parse_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of JSON objects."""
    entries: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entries.append(obj)
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON
                continue
    return entries


def parse_json(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSON file that contains either a list or dict of entries."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # If the file contains a dict, return its values
    if isinstance(data, dict):
        return list(data.values())
    return []


def extract_entry(record: Dict[str, Any], prompt_key: str, response_key: str, score_key: str) -> Optional[Dict[str, Any]]:
    """
    Extract a single feedback entry from an arbitrary log record.

    Args:
        record: The JSON object representing a single log entry.
        prompt_key: Name of the key containing the prompt/user message.
        response_key: Name of the key containing the assistant response.
        score_key: Name of the key containing the human-provided score or rating.

    Returns:
        A dict with prompt, response, and score fields, or None if any
        required field is missing or invalid.
    """
    prompt = record.get(prompt_key)
    response = record.get(response_key)
    score = record.get(score_key)
    # Allow nested structures for messages
    if prompt is None and "prompt" in record:
        prompt = record.get("prompt")
    if response is None and "response" in record:
        response = record.get("response")
    if score is None and ("score" in record or "rating" in record):
        score = record.get("score") or record.get("rating")

    if prompt is None or response is None or score is None:
        return None
    # Ensure the score is numeric; convert booleans or strings if possible
    if isinstance(score, bool):
        score = 1.0 if score else 0.0
    elif isinstance(score, str):
        try:
            score = float(score)
        except ValueError:
            return None
    return {"prompt": str(prompt), "response": str(response), "score": float(score)}


def collect_feedback(input_dir: str, prompt_key: str, response_key: str, score_key: str) -> List[Dict[str, Any]]:
    """
    Traverse a directory and collect feedback records from JSON/JSONL files.

    Args:
        input_dir: Directory containing log files.
        prompt_key: Key for the user prompt in the logs.
        response_key: Key for the model response.
        score_key: Key for the human score or rating.

    Returns:
        A list of feedback records (prompt/response/score).
    """
    dataset: List[Dict[str, Any]] = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(".jsonl"):
                path = os.path.join(root, fname)
                records = parse_jsonl(path)
            elif fname.lower().endswith(".json"):
                path = os.path.join(root, fname)
                records = parse_json(path)
            else:
                continue
            for rec in records:
                entry = extract_entry(rec, prompt_key, response_key, score_key)
                if entry is not None:
                    dataset.append(entry)
    return dataset


def write_jsonl(records: List[Dict[str, Any]], output_file: str) -> None:
    """Write a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect human feedback for reward model training.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSON or JSONL log files with human feedback.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output JSONL file to write consolidated feedback.",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt",
        help="Key used in the log files for the user prompt (default: 'prompt').",
    )
    parser.add_argument(
        "--response-key",
        type=str,
        default="response",
        help="Key used in the log files for the assistant response (default: 'response').",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="score",
        help="Key used in the log files for the human rating (default: 'score').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Collecting feedback from {args.input_dir}...")
    records = collect_feedback(
        input_dir=args.input_dir,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        score_key=args.score_key,
    )
    if not records:
        print("No valid feedback entries were found.")
    else:
        print(f"Collected {len(records)} feedback entries. Writing to {args.output_file}...")
        write_jsonl(records, args.output_file)
        print("Finished writing feedback dataset.")


if __name__ == "__main__":
    main()
