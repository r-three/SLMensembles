import torch
import random
import os
import shutil
import datasets
from transformers import AutoTokenizer
import simple_config as config

# ----------------------------------
# Helper Functions
# ----------------------------------
def compute_assistant_prefix_ids(tokenizer):
    """
    Automatically derive the assistant 'header' token IDs from the tokenizer's chat template.
    Works across OLMo-2 (<|assistant|>\n), Llama3.1, and other chat templates.
    """
    msgs = [{"role": "user", "content": "hi"}]
    wp = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    wop = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )
    # Normalize to flat lists (handle tensor vs list return types)
    wp = wp[0].tolist() if hasattr(wp, "shape") else wp
    wop = wop[0].tolist() if hasattr(wop, "shape") else wop
    prefix = wp[len(wop):]
    if not prefix:
        print("WARNING: assistant prefix is empty—chat template may not add a header for the assistant.")
    return prefix


def convert_conversations_to_messages(sample):
    """Convert Infinity-Instruct format to standard messages format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in sample["conversations"]:
        role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
        messages.append({"role": role, "content": turn.get("value", "")})
    sample["messages"] = messages
    return sample


def is_single_turn_qa(sample):
    """Keep only [human, gpt] two-turn items to keep label logic simple."""
    conv = sample.get("conversations", [])
    if len(conv) != 2:
        return False
    return conv[0].get("from") == "human" and conv[1].get("from") == "gpt"


def create_response_labels(sample, assistant_prefix_ids):
    """Mask everything except the assistant answer."""
    if not isinstance(sample["input_ids"], torch.Tensor):
        sample["input_ids"] = torch.tensor(sample["input_ids"], dtype=torch.long)

    if not isinstance(sample["attention_mask"], torch.Tensor):
        sample["attention_mask"] = torch.tensor(sample["attention_mask"], dtype=torch.long)

    input_ids = sample["input_ids"]
    attn = sample["attention_mask"]
    labels = input_ids.clone()
    labels.fill_(-100)

    # Find the first occurrence of the assistant prefix (computed generically)
    response_ids = assistant_prefix_ids
    start_pos = -1
    L = len(response_ids)
    
    # Scan only over valid portion (mask==1)
    valid_len = attn.nonzero(as_tuple=True)[0].max().item() + 1 if attn.sum() > 0 else len(input_ids)
    seq = input_ids[:valid_len].tolist()
    
    for i in range(0, valid_len - L + 1):
        if seq[i:i + L] == response_ids:
            start_pos = i + L
            break
    
    if start_pos == -1:
        # No assistant header found -> keep all -100 (filtered later)
        return labels
    
    end_pos = valid_len
    labels[start_pos:end_pos] = input_ids[start_pos:end_pos]
    labels = labels.masked_fill(attn == 0, -100)

    return labels

def format_chat_data(sample):
    """Format single sample with chat template (non-batched)."""
    return {"chat_text": tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False
    )}


def format_chat_data_batched(batch):
    """Format batch of samples with chat template (batched for speed)."""
    return {"chat_text": [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in batch["messages"]
    ]}


def tokenize_text_batched(batch):
    """Tokenize batch of chat texts (batched for speed)."""
    max_len = getattr(config, "max_length", 1024)
    outs = tokenizer(
        batch["chat_text"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    return {
        "input_ids": [x for x in outs["input_ids"]],
        "attention_mask": [x for x in outs["attention_mask"]],
    }


def contains_assistant_prefix(sample, assistant_prefix_ids):
    """Check if the example contains the assistant prefix in the valid region."""
    ids = sample["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    attn = sample["attention_mask"]
    if isinstance(attn, torch.Tensor):
        attn = attn.tolist()
    
    valid_len = max(i for i, m in enumerate(attn) if m == 1) + 1 if any(attn) else len(ids)
    window = ids[:valid_len]
    L = len(assistant_prefix_ids)
    
    for i in range(0, valid_len - L + 1):
        if window[i:i + L] == assistant_prefix_ids:
            return True
    return False


# ----------------------------------
# Load Dataset
# ----------------------------------
print("\n=== LOADING TOKENIZER ===")

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
print(f"Tokenizer loaded: {config.tokenizer_name}")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# Ensure pad token is set (OLMo and some models use eos_token as pad_token)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# Verify this is OLMo tokenizer (should be ~100k tokens)
if len(tokenizer) < 90000 or len(tokenizer) > 110000:
    print(f"   WARNING: Tokenizer vocab size ({len(tokenizer)}) seems unusual for OLMo!")
    print(f"   Expected ~100,278 tokens for OLMo-2 models")
    print(f"   Are you sure you're using the right tokenizer?")
else:
    print(f"✓ Tokenizer vocab size looks correct for OLMo")

# Automatically compute assistant prefix IDs from tokenizer
assistant_prefix_ids = compute_assistant_prefix_ids(tokenizer)
print(f"Assistant prefix token IDs: {assistant_prefix_ids}")

# ----------------------------------
# Load Dataset
# ----------------------------------
print("\n=== LOADING DATASET (Infinity-Instruct Gen) ===")
dataset = datasets.load_dataset(config.dataset_name, 'Gen', split="train")

print(f"Original dataset size: {len(dataset)}")
print(f"Original dataset features: {dataset.features}")

print(f"Example raw conversation format:")
ex = dataset[random.randint(0, len(dataset) - 1)]
print(f"Sample keys: {list(ex.keys())}")
print(f"Sample conversation (first turn): {ex['conversations'][0]}")

# ----------------------------------
# Filter to Single-Turn Q&A
# ----------------------------------
print("\n=== FILTERING TO SINGLE-TURN Q&A ===")
dataset = dataset.filter(is_single_turn_qa, num_proc=32)
print(f"After single-turn filter: {len(dataset)}")

# ----------------------------------
# Shuffle and Sample Dataset
# ----------------------------------
print("\n=== SHUFFLING AND SAMPLING ===")
dataset = dataset.shuffle(config.seed)
dataset = dataset.select(range(min(200_000, len(dataset))))
dataset = dataset.train_test_split(test_size=min(2000, max(10, len(dataset)//50)))
print(f"Splits -> Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")

# ------------------------------------------
# Convert conversations to messages format
# ------------------------------------------
print("\n=== CONVERTING TO MESSAGES FORMAT ===")
dataset = dataset.map(convert_conversations_to_messages, num_proc=32)

# ------------------------------------------
# Apply preprocessing to format chat data (batched for speed)
# ------------------------------------------
print("\n=== APPLYING CHAT TEMPLATE ===")
processed_dataset = dataset.map(format_chat_data_batched, batched=True, num_proc=32)

print(f"Examples after chat formatting:")
print(f"Train example chat_text (first 300 chars):\n{processed_dataset['train'][0]['chat_text'][:300]}...")
print(f"Test example chat_text (first 300 chars):\n{processed_dataset['test'][0]['chat_text'][:300]}...")

# --------------------------
# Tokenize the text (batched for speed)
# --------------------------
print("\n=== TOKENIZING TEXT ===")
# Safer column removal - only remove columns that exist
cols_to_remove = [c for c in ["messages", "conversations", "label", "langdetect", "source"] 
                  if c in processed_dataset["train"].column_names]
tokenized_dataset = processed_dataset.map(
    tokenize_text_batched, 
    remove_columns=cols_to_remove, 
    batched=True, 
    num_proc=32
)
print(f"Dataset features after tokenization: {tokenized_dataset['train'].features}")

print(f"Train example input_ids shape: {torch.tensor(tokenized_dataset['train'][0]['input_ids']).shape}")
print(f"Train example attention_mask shape: {torch.tensor(tokenized_dataset['train'][0]['attention_mask']).shape}")
print(f"Train example id: {tokenized_dataset['train'][0]['id']}")

print("\n=== ADDING LABELS (assistant only) ===")
labeled_dataset = tokenized_dataset.map(
    lambda s: {"labels": create_response_labels(s, assistant_prefix_ids)}, 
    num_proc=32
)
print(f"Dataset features after adding labels: {labeled_dataset['train'].features}")
print(f"ID column preserved - example id: {labeled_dataset['train'][0]['id']}")

# -----------------------------------------
# Filter out samples which were truncated
# -----------------------------------------
print("\n=== FILTERING (must contain assistant header in valid region) ===")

labeled_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "id"])
final_dataset = labeled_dataset.filter(
    lambda s: contains_assistant_prefix(s, assistant_prefix_ids), 
    num_proc=32
)
print(f"Kept -> Train: {len(final_dataset['train'])} | Test: {len(final_dataset['test'])}")

# ------------------------------
# Save the processed dataset
# ------------------------------
print("\n=== SAVING DATASET ===")

save_path = config.dataset_path
if os.path.exists(save_path):
    shutil.rmtree(save_path) 
final_dataset.save_to_disk(save_path)
print(f"Dataset saved to: {save_path}")

# ------ Save a clean version with only required columns for training ------
clean_cols = [c for c in ["chat_text"] if c in final_dataset["train"].column_names]
clean_dataset = final_dataset.remove_columns(clean_cols) if clean_cols else final_dataset
clean_save_path = save_path + "_clean"
if os.path.exists(clean_save_path):
    shutil.rmtree(clean_save_path)
clean_dataset.save_to_disk(clean_save_path)
print(f"Clean dataset saved to: {clean_save_path}")
# ------- End saving code ------------

# ------ Verify token IDs are within vocabulary range ------
print("\n=== VERIFYING TOKEN IDs ===")

max_id = 0
min_id = 10**9
sample_n = min(1000, len(final_dataset["train"]))
print(f"Checking token ID ranges in first {sample_n} samples...")

for ex in final_dataset["train"].select(range(sample_n)):
    ii = ex["input_ids"].tolist() if isinstance(ex["input_ids"], torch.Tensor) else ex["input_ids"]
    if ii:
        max_id = max(max_id, max(ii))
        min_id = min(min_id, min(ii))

print(f"Token ID range: {min_id} to {max_id}")
print(f"Tokenizer vocab size: {len(tokenizer)}")

if max_id >= len(tokenizer):
    print(f"❌ ERROR: Max token ID ({max_id}) >= vocab size ({len(tokenizer)})")
    print(f"   This will cause CUDA errors during training!")
    print(f"   Check that you're using the correct tokenizer.")
else:
    print(f"✓ All token IDs are within vocabulary range!")

print("\n✓ Dataset processing complete!")

