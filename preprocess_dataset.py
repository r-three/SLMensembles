import datasets
from transformers import AutoTokenizer
import numpy as np


seed = 42
tokenizer_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_name = "allenai/tulu-3-sft-mixture"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
dataset = datasets.load_dataset(dataset_name, split="train")
dataset = dataset.shuffle(seed)
dataset = dataset.select(range(200_000))
dataset = dataset.train_test_split(test_size=2000)

def format_and_tokenize(example):
    chat_prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    tokenized = tokenizer(chat_prompt, truncation=True, padding="max_length", max_length=1024, return_tensors="pt")
    return {
        "input_ids": tokenized["input_ids"].squeeze(0),
        "attention_mask": tokenized["attention_mask"].squeeze(0)
    }

tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=["messages", "id", "source"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
def filter_truncated_sequences(example):
    for start_idx in np.where(example["input_ids"] == response_template_ids[0])[0]:
        if response_template_ids == example["input_ids"][start_idx : start_idx + len(response_template_ids)].tolist():
            return True
    return False

tokenized_dataset = tokenized_dataset.filter(filter_truncated_sequences)
tokenized_dataset.save_to_disk("/scratch/ssd004/scratch/nkandpa2/slm_ensembles/tulu-3-sft-mixture-pretokenized")
