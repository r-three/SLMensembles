import torch
import numpy as np
import random
import datasets
from transformers import AutoTokenizer
from config import seed, dataset_name, tokenizer_name


def create_response_labels(input_ids):
    """
    Creates labels for causal language modeling that masks everything except the assistant's response.

    Args:
        input_ids: Tensor containing token IDs
        tokenizer: The tokenizer used to identify assistant template markers

    Returns:
        Tensor of same shape as input_ids with -100 for masked tokens
    """
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)

    labels = input_ids.clone()  # Clone input_ids to create labels
    response_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]  # Get the assistant response template IDs

    # By default, mask everything with -100
    labels.fill_(-100)

    # Find where the assistant response starts
    start_pos = -1
    for i in range(len(input_ids) - len(response_ids) + 1):
        if input_ids[i : i + len(response_ids)].tolist() == response_ids:
            # We want tokens after the template marker
            start_pos = i + len(response_ids)
            break

    # If we found the start position, unmask those tokens
    if start_pos != -1:
        labels[start_pos:] = input_ids[start_pos:]

    return labels


def format_chat_data(sample):
    """
    Formats chat data using the tokenizer's chat template.
    """
    return {"chat_text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}


def tokenize_text(sample):
    """
    Tokenizes text data with appropriate padding and truncation.
    """
    tokenized = tokenizer(sample["chat_text"], truncation=True, padding="max_length", max_length=1024, return_tensors="pt")

    return {"input_ids": tokenized["input_ids"].squeeze(0), "attention_mask": tokenized["attention_mask"].squeeze(0)}


def add_labels(sample):
    """
    Adds properly masked labels for assistant responses.
    """
    sample["labels"] = create_response_labels(sample["input_ids"])
    return sample


# Path to your saved dataset
dataset_path = "/scratch/ssd004/scratch/klambert/slm_ensembles/tulu-3-sft-mixture-pretokenized"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Load the tulu-3-sft-mixture dataset (a SFT dataset from Allen AI), shuffles it, selects 200,000 examples, and splits it into train/test sets

print("\n=== LOADING DATASET ===")
dataset = datasets.load_dataset(dataset_name, split="train")
print(f"Original dataset size: {len(dataset)}")
print(f"Original dataset features: {dataset.features}")
print(f"Example raw message format:")
random_idx = random.randint(0, len(dataset) - 1)
print(dataset[random_idx]["messages"])
print(f"Another example raw message format:")
random_idx = random.randint(0, len(dataset) - 1)
print(dataset[random_idx]["messages"])

# Shuffle and sample the dataset
dataset = dataset.shuffle(seed)
dataset = dataset.select(range(200_000))
dataset = dataset.train_test_split(test_size=2000)
print(f"\nAfter sampling - Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

# TODO: split into the train, validation, and test sets

# Apply preprocessing to format chat data
print("\n=== APPLYING CHAT TEMPLATE ===")
processed_dataset = dataset.map(format_chat_data)
print(f"Examples after chat formatting:")
print(f"Train example chat_text (first 300 chars):\n{processed_dataset['train'][0]['chat_text'][:300]}...")
print(f"Test example chat_text (first 300 chars):\n{processed_dataset['test'][0]['chat_text'][:300]}...")

# Tokenize the text
print("\n=== TOKENIZING TEXT ===")
tokenized_dataset = processed_dataset.map(tokenize_text, remove_columns=["messages", "id", "source"])
print(f"Dataset features after tokenization: {tokenized_dataset['train'].features}")
print(f"Train example input_ids shape: {torch.tensor(tokenized_dataset['train'][0]['input_ids']).shape}")
print(f"Train example attention_mask shape: {torch.tensor(tokenized_dataset['train'][0]['attention_mask']).shape}")


print("\n=== ADDING LABELS ===")
labeled_dataset = tokenized_dataset.map(add_labels)
print(f"Dataset features after adding labels: {labeled_dataset['train'].features}")

# Set format for PyTorch
labeled_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Filter out samples which were truncated
print("\n=== FILTERING EXAMPLES ===")


def contains_complete_response_template(sample):
    """Check if the example contains the complete assistant response template."""
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]

    for start_idx in range(len(sample["input_ids"]) - len(response_template_ids) + 1):
        if sample["input_ids"][start_idx : start_idx + len(response_template_ids)].tolist() == response_template_ids:
            return True
    return False


# Check how many examples will be filtered out
num_train_before = len(labeled_dataset["train"])
train_keep_count = sum(1 for _ in filter(lambda x: contains_complete_response_template(x), (labeled_dataset["train"][i] for i in range(min(1000, num_train_before)))))
print(f"Estimated percentage of train examples to keep: {train_keep_count/min(1000, num_train_before)*100:.2f}% (based on 1000 samples)")

# Apply the filter
final_dataset = labeled_dataset.filter(contains_complete_response_template)
print(f"Dataset size after filtering - Train: {len(final_dataset['train'])}, Test: {len(final_dataset['test'])}")

# Save the processed dataset
print("\n=== SAVING DATASET ===")
save_path = "/scratch/ssd004/scratch/klambert/slm_ensembles/tulu-3-sft-mixture-pretokenized"
final_dataset.save_to_disk(save_path)
print(f"Dataset saved to: {save_path}")
print("Dataset processing complete!")
