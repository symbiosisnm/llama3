"""
Tokenizer training script to build a Byte‑Level BPE tokenizer from business data and integrate it with a pre‑trained language model.

This script trains a tokenizer using the HuggingFace tokenizers library on a
collection of raw text files (such as cleaned prompts and responses). It also
resizes the embedding layer of a base language model to match the new
vocabulary and saves both the tokenizer and model.
"""

import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer  # type: ignore
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM


def train_tokenizer(
    train_files: list[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: list[str],
    output_dir: str,
) -> PreTrainedTokenizerFast:
    """Train a Byte‑Level BPE tokenizer and wrap it as a fast tokenizer.

    Args:
        train_files: List of paths to plain‑text files used for training.
        vocab_size: Number of tokens in the vocabulary.
        min_frequency: Minimum token frequency to be included.
        special_tokens: List of special tokens (e.g., BOS, EOS, PAD, UNK, MASK).
        output_dir: Directory to save the tokenizer files.

    Returns:
        A PreTrainedTokenizerFast instance.
    """
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=train_files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Save the raw tokenizer files (vocab and merges)
    tokenizer.save_model(output_dir)
    # Wrap the tokenizer as a PreTrainedTokenizerFast
    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(output_dir) / "tokenizer.json"),
        unk_token=special_tokens[3],
        bos_token=special_tokens[0],
        eos_token=special_tokens[1],
        pad_token=special_tokens[2],
        mask_token=special_tokens[4] if len(special_tokens) > 4 else None,
    )
    # Save the fast tokenizer config
    tokenizer_fast.save_pretrained(output_dir)
    return tokenizer_fast


def integrate_tokenizer_with_model(
    tokenizer_dir: str, base_model_name: str, output_dir: str
) -> None:
    """Resize a base language model to match the new tokenizer vocabulary.

    Args:
        tokenizer_dir: Directory containing the trained tokenizer.
        base_model_name: Name or path of the base causal language model.
        output_dir: Directory to save the resized model and tokenizer.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    # Resize token embeddings to match the new vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a ByteLevel BPE tokenizer on business data and integrate it "
            "with a base language model."
        )
    )
    parser.add_argument(
        "--train_files",
        nargs="+",
        required=True,
        help="List of text files to train the tokenizer.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=52000,
        help="Vocabulary size for the tokenizer (default: 52000).",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens to be included (default: 2).",
    )
    parser.add_argument(
        "--base_model_name",
        required=True,
        help="Base model name or path for integration (e.g., meta-llama/Meta-Llama-3-8B).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the trained tokenizer and resized model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    print(f"Training tokenizer on files: {args.train_files}")
    tokenizer = train_tokenizer(
        args.train_files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        output_dir=args.output_dir,
    )
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Integrating tokenizer with model {args.base_model_name}")
    integrate_tokenizer_with_model(
        tokenizer_dir=args.output_dir,
        base_model_name=args.base_model_name,
        output_dir=args.output_dir,
    )
    print(f"Saved tokenizer and resized model to {args.output_dir}")


if __name__ == "__main__":
    main()
