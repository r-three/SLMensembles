import os
import random
import shutil
from typing import List, Dict, Any

import torch
import datasets
from transformers import AutoTokenizer
import torch.nn.functional as F

import config  # unchanged

# -----------------------------
# Globals set after tokenizer load
# -----------------------------
ASSISTANT_PREFIX_IDS: List[int] = []

# -----------------------------
# Utilities
# -----------------------------
def map_conversations_to_messages(conv: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Infinity-Instruct -> HF chat template message format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    msgs = []
    for turn in conv:
        role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
        msgs.append({"role": role, "content": turn.get("value", "")})
    return msgs

def is_single_turn_qa(sample: Dict[str, Any]) -> bool:
    """Keep only [human, gpt] two-turn items to keep label logic simple."""
    conv = sample.get("conversations", [])
    if len(conv) != 2:
        return False
    return conv[0].get("from") == "human" and conv[1].get("from") == "gpt"

def compute_assistant_prefix_ids(tokenizer) -> List[int]:
    """
    Derive the assistant 'header' token ids for THIS tokenizer's chat template.
    Works across OLMo-2 (<|assistant|>\n), Llama3.1 chatml, etc.
    """
    msgs = [{"role": "user", "content": "hi"}]
    with_prompt = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors=None
    )["input_ids"]
    without_prompt = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False, return_tensors=None
    )["input_ids"]
    # tail that was appended by add_generation_prompt=True is the assistant header
    prefix = with_prompt[0][len(without_prompt[0]):]
    return prefix

# -----------------------------
# Labeling helpers
# -----------------------------
def create_response_labels(sample: Dict[str, Any], tokenizer) -> torch.Tensor:
    """Mask everything except the assistant answer (single-turn)."""
    if not isinstance(sample["input_ids"], torch.Tensor):
        sample["input_ids"] = torch.tensor(sample["input_ids"], dtype=torch.long)
    if not isinstance(sample["attention_mask"], torch.Tensor):
        sample["attention_mask"] = torch.tensor(sample["attention_mask"], dtype=torch.long)

    input_ids = sample["input_ids"]
    attn = sample["attention_mask"]
    labels = input_ids.clone()
    labels.fill_(-100)

    # Find the first occurrence of the assistant prefix (computed generically)
    global ASSISTANT_PREFIX_IDS
    response_ids = ASSISTANT_PREFIX_IDS
    start_pos = -1
    L = len(response_ids)
    # scan only over valid portion (mask==1)
    valid_len = attn.nonzero(as_tuple=True)[0].max().item() + 1
    seq = input_ids[:valid_len].tolist()
    for i in range(0, valid_len - L + 1):
        if seq[i:i + L] == response_ids:
            start_pos = i + L
            break

    if start_pos == -1:
        # no assistant header found -> keep all -100 (filtered later)
        return labels

    end_pos = valid_len
    labels[start_pos:end_pos] = input_ids[start_pos:end_pos]
    labels = labels.masked_fill(attn == 0, -100)
    return labels

def contains_assistant_prefix(sample: Dict[str, Any]) -> bool:
    """Used to filter out truncated samples with missing assistant header."""
    ids = sample["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    attn = sample["attention_mask"]
    if isinstance(attn, torch.Tensor):
        attn = attn.tolist()
    valid_len = max(i for i, m in enumerate(attn) if m == 1) + 1 if any(attn) else len(ids)
    window = ids[:valid_len]
    L = len(ASSISTANT_PREFIX_IDS)
    for i in range(0, valid_len - L + 1):
        if window[i:i + L] == ASSISTANT_PREFIX_IDS:
            return True
    return False

# -----------------------------
# Mapping / tokenization
# -----------------------------
def format_chat_text(sample: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    msgs = map_conversations_to_messages(sample["conversations"])
    sample["messages"] = msgs  # optional, in case you want to inspect later
    sample["chat_text"] = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    return sample

def tokenize_text(sample: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    tokenized = tokenizer(
        sample["chat_text"],
        truncation=True,
        padding="max_length",
        max_length=getattr(config, "max_length", 1024),
        return_tensors="pt",
    )
    sample["input_ids"] = tokenized["input_ids"].squeeze(0)
    sample["attention_mask"] = tokenized["attention_mask"].squeeze(0)
    return sample

# -----------------------------
# Main
# -----------------------------
print("\n=== LOADING TOKENIZER ===")
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
print(f"Tokenizer: {config.tokenizer_name}  |  vocab={len(tokenizer)}")

ASSISTANT_PREFIX_IDS = compute_assistant_prefix_ids(tokenizer)
print(f"Assistant prefix token IDs: {ASSISTANT_PREFIX_IDS}")

print("\n=== LOADING DATASET (Infinity-Instruct Gen) ===")
# Support optional subset name in your config, default to 'Gen'
subset = getattr(config, "dataset_subset", "Gen")
# BAAI/Infinity-Instruct requires a config name like 'Gen' or '7M'
dataset = datasets.load_dataset(config.dataset_name, subset, split="train")  # e.g., name='BAAI/Infinity-Instruct'
print(f"Raw size: {len(dataset)} | features: {dataset.features}")

# Quick peek
ex = dataset[random.randint(0, len(dataset)-1)]
print("Sample keys:", list(ex.keys()))
print("Sample conv (first turn):", ex["conversations"][0])

# Restrict to clean, single-turn Q/A
dataset = dataset.filter(is_single_turn_qa, num_proc=32)
print(f"After single-turn filter: {len(dataset)}")

# Shuffle & sample (keep same limits as before)
dataset = dataset.shuffle(config.seed)
dataset = dataset.select(range(min(200_000, len(dataset))))
dataset = dataset.train_test_split(test_size=min(2000, max(10, len(dataset)//50)))
print(f"Splits -> train: {len(dataset['train'])} | test: {len(dataset['test'])}")

print("\n=== APPLYING CHAT TEMPLATE ===")
processed = dataset.map(lambda s: format_chat_text(s, tokenizer), num_proc=32)

print("\n=== TOKENIZING ===")
cols_to_remove = [c for c in ["conversations", "messages", "label", "langdetect", "source"] if c in processed["train"].column_names]
tokenized = processed.map(lambda s: tokenize_text(s, tokenizer), remove_columns=cols_to_remove, num_proc=32)

print("\n=== ADDING LABELS (assistant only) ===")
labeled = tokenized.map(lambda s: {"labels": create_response_labels(s, tokenizer)}, num_proc=32)

# Torch format for a quick sample-based retention estimate
labeled.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("\n=== FILTERING (must contain assistant header in valid region) ===")
final_ds = labeled.filter(contains_assistant_prefix, num_proc=32)
print(f"Kept -> train: {len(final_ds['train'])} | test: {len(final_ds['test'])}")

print("\n=== SAVING ===")
save_path = config.dataset_path
if os.path.exists(save_path):
    shutil.rmtree(save_path)
final_ds.save_to_disk(save_path)
print(f"Saved: {save_path}")

clean_ds = final_ds.remove_columns([c for c in ["chat_text"] if c in final_ds["train"].column_names])
clean_path = save_path + "_clean"
if os.path.exists(clean_path):
    shutil.rmtree(clean_path)
clean_ds.save_to_disk(clean_path)
print(f"Saved (clean): {clean_path}")

print("\n=== VERIFYING TOKEN IDS (first 1k) ===")
max_id = 0
min_id = 10**9
sample_n = min(1000, len(final_ds["train"]))
for ex in final_ds["train"].select(range(sample_n)):
    ii = ex["input_ids"].tolist() if isinstance(ex["input_ids"], torch.Tensor) else ex["input_ids"]
    if ii:
        max_id = max(max_id, max(ii))
        min_id = min(min_id, min(ii))
print(f"Range: {min_id} .. {max_id} | vocab={len(tokenizer)}")
if max_id >= len(tokenizer):
    print("❌ ERROR: token id >= vocab size. Check tokenizer/dataset mismatch.")
else:
    print("✓ IDs within vocab range.")
print("\nDone.")
