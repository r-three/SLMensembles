import torch
import random
import datasets
from transformers import AutoTokenizer
import config
import torch.nn.functional as F

# ----------------------------------
# Helper Functions
# ----------------------------------
def create_response_labels(input_ids):
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)

    labels = input_ids.clone()
    response_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    labels.fill_(-100)

    start_pos = -1
    for i in range(len(input_ids) - len(response_ids) + 1):
        if input_ids[i : i + len(response_ids)].tolist() == response_ids:
            start_pos = i + len(response_ids)
            break

    if start_pos != -1:
        labels[start_pos:] = input_ids[start_pos:]

    return labels

# TODO: subset the IDs 
# 1. take tulu 3
# 2. only keep the examples that match the ids in the clustered dataset (from Malikeh's dataset - filter the ids)

# TODO: add the ids column to this script
# https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
# subset dataset name from the clustered dataset - pass as argument when loading 
# use the filter functoin that goes over the original dataset and checks if the ids are present in the subset 
# transplant the ids into the logic cached dataset  (either do before - transplant the ids; but do some checking on the ordering and that ti's consistent
# # or after, where the subsetting filtering would be run on the logit cached)

def format_chat_data(sample):
    return {"chat_text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}


def tokenize_text(sample):
    tokenized = tokenizer(
        sample["chat_text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",
    )

    return {
        "input_ids": tokenized["input_ids"].squeeze(0),
        "attention_mask": tokenized["attention_mask"].squeeze(0),
    }


def add_labels(sample):
    sample["labels"] = create_response_labels(sample["input_ids"])
    return sample


def contains_complete_response_template(sample):
    """Check if the example contains the complete assistant response template."""
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]

    for start_idx in range(len(sample["input_ids"]) - len(response_template_ids) + 1):
        if sample["input_ids"][start_idx : start_idx + len(response_template_ids)].tolist() == response_template_ids:
            return True
    return False


# ----------------------------------
# Load Dataset
# ----------------------------------
print("\n=== LOADING DATASET ===")

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
dataset = datasets.load_dataset(config.dataset_name, split="train")

print(f"Original dataset size: {len(dataset)}")
print(f"Original dataset features: {dataset.features}")

print(f"Example raw message format:")
print(dataset[random.randint(0, len(dataset) - 1)]["messages"])
print(f"Another example raw message format:")
print(dataset[random.randint(0, len(dataset) - 1)]["messages"])

# ----------------------------------
# Shuffle and Sample Dataset
# ----------------------------------
dataset = dataset.shuffle(config.seed)
dataset = dataset.select(range(200_000))
dataset = dataset.train_test_split(test_size=2000)
print(f"\nAfter sampling - Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

# ------------------------------------------
# Apply preprocessing to format chat data
# ------------------------------------------
print("\n=== APPLYING CHAT TEMPLATE ===")
processed_dataset = dataset.map(format_chat_data, num_proc=8)

print(f"Examples after chat formatting:")
print(f"Train example chat_text (first 300 chars):\n{processed_dataset['train'][0]['chat_text'][:300]}...")
print(f"Test example chat_text (first 300 chars):\n{processed_dataset['test'][0]['chat_text'][:300]}...")

# --------------------------
# Tokenize the text
# --------------------------
print("\n=== TOKENIZING TEXT ===")
tokenized_dataset = processed_dataset.map(tokenize_text, remove_columns=["messages", "source"], num_proc=8)
print(f"Dataset features after tokenization: {tokenized_dataset['train'].features}")

print(f"Train example input_ids shape: {torch.tensor(tokenized_dataset['train'][0]['input_ids']).shape}")
print(f"Train example attention_mask shape: {torch.tensor(tokenized_dataset['train'][0]['attention_mask']).shape}")
print(f"Train example id: {tokenized_dataset['train'][0]['id']}")

labeled_dataset = tokenized_dataset.map(add_labels, num_proc=8)
print(f"Dataset features after adding labels: {labeled_dataset['train'].features}")
print(f"ID column preserved - example id: {labeled_dataset['train'][0]['id']}")

# -----------------------------------------
# Filter out samples which were truncated
# -----------------------------------------
print("\n=== FILTERING EXAMPLES ===")

labeled_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
num_train_before = len(labeled_dataset["train"])
train_keep_count = sum(
    1
    for _ in filter(
        lambda x: contains_complete_response_template(x),
        (labeled_dataset["train"][i] for i in range(min(1000, num_train_before))),
    )
)
print(
    f"Estimated percentage of train examples to keep: {train_keep_count/min(1000, num_train_before)*100:.2f}% (based on 1000 samples)"
)

final_dataset = labeled_dataset.filter(contains_complete_response_template, num_proc=8)
print(f"Dataset size after filtering - Train: {len(final_dataset['train'])}, Test: {len(final_dataset['test'])}")

# ------------------------------
# Save the processed dataset
# ------------------------------
print("\n=== SAVING DATASET ===")

save_path = config.dataset_path
final_dataset.save_to_disk(save_path)
print(f"Dataset saved to: {save_path}")
print("Dataset processing complete!")
