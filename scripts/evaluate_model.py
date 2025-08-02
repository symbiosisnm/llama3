"""
Evaluation script for business LLMs.

This script evaluates a fine‑tuned or RLHF‑trained language model on a dataset
of prompt/response pairs. It generates responses using the model and
calculates evaluation metrics such as BLEU and ROUGE. Perplexity can also
optionally be computed if supported by the evaluate library.

Example usage:

    python scripts/evaluate_model.py \
        --model_path ./llama3-business-finetune \
        --data_path ./eval_data.jsonl \
        --max_samples 100 \
        --max_new_tokens 256
"""

import argparse
from typing import List

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the model given a prompt.

    Args:
        model: Pretrained causal language model.
        tokenizer: Corresponding tokenizer.
        prompt: Prompt string to feed to the model.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        Generated response string.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # Remove the prompt portion from the generated sequence
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text[len(prompt) :].strip()


def compute_metrics(predictions: List[str], references: List[str]) -> dict:
    """Compute BLEU and ROUGE metrics.

    Args:
        predictions: List of generated responses.
        references: List of reference responses.

    Returns:
        Dictionary of metric results.
    """
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    # BLEU expects references as list of lists
    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_result = rouge.compute(predictions=predictions, references=references)
    return {"bleu": bleu_result, "rouge": rouge_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine‑tuned language model on a dataset")
    parser.add_argument(
        "--model_path", required=True, help="Path to the fine‑tuned or RLHF model directory"
    )
    parser.add_argument(
        "--data_path", required=True, help="Path to JSONL file containing eval data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (useful for quick tests)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate for each prompt",
    )
    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load evaluation dataset
    dataset = load_dataset("json", data_files={"data": args.data_path})["data"]
    if args.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    predictions: List[str] = []
    references: List[str] = []

    for example in dataset:
        user_prompt = example.get("prompt")
        reference = example.get("response")
        prompt = f"User: {user_prompt}\nAssistant:"
        generated = generate_response(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
        predictions.append(generated)
        references.append(reference)

    # Compute evaluation metrics
    metrics = compute_metrics(predictions, references)
    print("Evaluation Results:")
    print(metrics)


if __name__ == "__main__":
    main()
