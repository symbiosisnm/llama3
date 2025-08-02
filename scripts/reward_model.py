import os
import argparse

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, mean_squared_error


def preprocess(example):
    """
    Combine prompt and response into a single text field for reward model training.
    """
    text = f"User: {example['prompt']}\nAssistant: {example['response']}"
    example['text'] = text
    return example


def compute_metrics_fn(num_labels):
    """
    Create a metrics function for classification (accuracy) or regression (MSE).
    """
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if num_labels == 1:
            preds = logits.squeeze()
            mse = mean_squared_error(labels, preds)
            return {'mse': mse}
        else:
            preds = np.argmax(logits, axis=1)
            acc = accuracy_score(labels, preds)
            return {'accuracy': acc}
    return compute_metrics


def train_reward_model(
    model_name: str,
    data_path: str,
    output_dir: str,
    num_labels: int = 1,
    num_epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    val_split: float = 0.1,
    max_length: int = 512,
):
    """
    Train a reward model on a dataset of prompt/response pairs with associated scores.
    Args:
        model_name: Base model to fine‑tune (e.g., 'bert-base-uncased').
        data_path: Path to JSON or JSONL file containing 'prompt', 'response', and 'score' fields.
        output_dir: Directory to save the trained reward model.
        num_labels: Number of labels (1 for regression, >1 for classification).
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for training.
        val_split: Fraction of data used for validation.
        max_length: Maximum tokenized sequence length.
    """
    dataset = load_dataset('json', data_files={'data': data_path})['data']
    # Split dataset into train and validation
    if val_split > 0:
        split = dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
    else:
        train_dataset = dataset
        eval_dataset = None

    # Preprocess to add combined text field
    train_dataset = train_dataset.map(preprocess)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(preprocess)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    train_dataset = train_dataset.rename_column('score', 'labels')
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
        eval_dataset = eval_dataset.rename_column('score', 'labels')
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy='epoch' if eval_dataset is not None else 'no',
        learning_rate=learning_rate,
        logging_dir=os.path.join(output_dir, 'logs'),
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model='mse' if num_labels == 1 else 'accuracy',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn(num_labels) if eval_dataset is not None else None,
    )

    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description='Train a reward model for RLHF.')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Base model name or path.')
    parser.add_argument('--data_path', type=str, required=True, help="Path to JSON/JSONL file with 'prompt', 'response', 'score' fields.")
    parser.add_argument('--output_dir', type=str, default='./reward_model', help='Directory to save the trained model.')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels (1 for regression, >1 for classification).')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data to use for validation.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization.')
    args = parser.parse_args()

    train_reward_model(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_labels=args.num_labels,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        max_length=args.max_length,
    )


if __name__ == '__main__':
    main()
