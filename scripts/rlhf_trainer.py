import argparse
import os
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig


def compute_rewards(reward_model, tokenizer, prompts: List[str], responses: List[str]):
    """
    Compute reward scores for prompt‑response pairs using a reward model.
    The reward model can be a classification model (num_labels > 1) or a regression model (num_labels = 1).
    """
    # Combine prompt and response into a single sequence for scoring
    texts = [f"{prompt} {response}" for prompt, response in zip(prompts, responses)]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = reward_model(**inputs)
    logits = outputs.logits
    # If regression head, logits is shape (batch_size, 1). Otherwise take probability of positive class.
    if reward_model.config.num_labels == 1:
        scores = logits.squeeze(-1)
    else:
        # use softmax to get probability of class 1
        probs = torch.nn.functional.softmax(logits, dim=-1)
        scores = probs[:, 1]
    # Convert scores to python list for PPOTrainer
    return scores.detach().cpu().tolist()


def train_rlhf(
    base_model_name: str,
    reward_model_name: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    max_new_tokens: int = 128,
):
    """
    Train a language model using Reinforcement Learning from Human Feedback (RLHF).
    Args:
        base_model_name: path or name of a pretrained SFT model to start from.
        reward_model_name: path or name of a reward model used to score responses.
        data_path: path to the JSON or JSONL file containing a 'prompt' field.
        output_dir: directory to save the RL‑tuned model.
        num_epochs: number of passes through the dataset.
        batch_size: number of prompts to process per PPO step.
        learning_rate: learning rate for PPO.
        max_new_tokens: maximum number of tokens to generate for each response.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    # Use a reference model for KL penalty
    ref_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
    reward_model.eval()

    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref_model.to(device)
    reward_model.to(device)

    # Load dataset containing prompts
    dataset = load_dataset("json", data_files={"train": data_path})["train"]
    prompts = dataset["prompt"]

    # Setup PPO configuration and trainer
    config = PPOConfig(batch_size=batch_size, learning_rate=learning_rate)
    ppo_trainer = PPOTrainer(model=model, ref_model=ref_model, tokenizer=tokenizer, config=config)

    # Training loop
    for epoch in range(num_epochs):
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]

            query_tensors = []
            response_tensors = []
            responses = []

            # Generate responses for each prompt in the batch
            for prompt in batch_prompts:
                # Encode the prompt and move to device
                query_input = tokenizer(prompt, return_tensors="pt").to(device)
                query_tensor = query_input["input_ids"][0]
                query_tensors.append(query_tensor)

                # Generate model output
                output = model.generate(
                    **query_input,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Extract the generated response tokens (after the prompt)
                response_tensor = output[0][query_tensor.shape[0] :]
                response_tensors.append(response_tensor)

                # Decode full sequence for reward scoring
                full_text = tokenizer.decode(output[0], skip_special_tokens=True)
                responses.append(full_text[len(prompt):].strip())

            # Compute reward scores
            rewards = compute_rewards(reward_model, tokenizer, batch_prompts, responses)

            # Perform PPO step
            ppo_trainer.step(query_tensors, response_tensors, rewards)

    # Save the fine‑tuned model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Train a language model with RLHF.")
    parser.add_argument("--base_model", type=str, required=True, help="Pretrained base/SFT model name or path.")
    parser.add_argument("--reward_model", type=str, required=True, help="Reward model name or path.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON/JSONL dataset with 'prompt' field.")
    parser.add_argument("--output_dir", type=str, default="./rlhf_output", help="Where to save the RLHF‑tuned model.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to run over the dataset.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for PPO.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for PPO.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate per response.")
    args = parser.parse_args()

    train_rlhf(
        base_model_name=args.base_model,
        reward_model_name=args.reward_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
