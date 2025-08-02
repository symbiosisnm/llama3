import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling


def preprocess_function(examples, tokenizer, max_length):
    # Combine prompt and response into a single text
    combined = [f"User: {p}\nAssistant: {r}" for p, r in zip(examples['prompt'], examples['response'])]
    tokenized = tokenizer(combined, truncation=True, padding='max_length', max_length=max_length)
    return tokenized


def train_sft(model_name: str, data_path: str, output_dir: str, batch_size: int = 2, epochs: int = 3, max_length: int = 1024):
    """Fine-tune a base LLM on business data using supervised fine-tuning (SFT)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Load dataset from JSONL file
    dataset = load_dataset("json", data_files={"train": data_path})["train"]
    # Preprocess dataset
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True, remove_columns=dataset.column_names)
    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=50,
        save_steps=500,
        fp16=True,
        evaluation_strategy="no"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    trainer.train()
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on business data using supervised fine-tuning.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Base model name or path")
    parser.add_argument("--data_path", type=str, default="data/business_data.jsonl", help="Path to JSONL training data")
    parser.add_argument("--output_dir", type=str, default="sft_output", help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    args = parser.parse_args()
    train_sft(args.model_name, args.data_path, args.output_dir, args.batch_size, args.epochs, args.max_length)


if __name__ == "__main__":
    main()
