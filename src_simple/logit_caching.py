import os
import torch
import glob
import shutil
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from simple_config import config
from simple_utils import get_dataset, prepare_dataset
from tqdm import tqdm
import math



def get_teacher_logprobs():
    """Load or generate teacher logits dataset."""
    if not os.path.exists(os.path.join(config.logprob_cache_path, "teacher_logprobs")):
        cache_teacher_logprobs()

    print("--> Loading Teacher Logits")
    dataset = load_from_disk(os.path.join(config.logprob_cache_path, "teacher_logprobs"))
    
    return dataset


def concatenate_logprobs_chunks(split_dirs: list[str]):
    """Concatenate logit chunks into a single dataset."""
    datasets_list = []
    for split_dir in split_dirs:
        chunk_paths = sorted(
            [p for p in glob.glob(os.path.join(split_dir, "chunk_*")) if os.path.isdir(p)],
            key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
        )
        print(f"--> Loading {len(chunk_paths)} chunks for '{os.path.basename(split_dir)}' split")
        datasets_list.extend(load_from_disk(p) for p in chunk_paths)
    return concatenate_datasets(datasets_list) if datasets_list else None


def build_teacher_logprobs_dataset():
    """Build the final teacher logits dataset from chunks."""
    data_dict = {}

    for split in ["train", "test"]:
        split_dirs = sorted(
            [d for d in glob.glob(os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}_*")) if os.path.isdir(d)]
        )
        split_ds = concatenate_logprobs_chunks(split_dirs)
        data_dict[split] = split_ds

    dataset = DatasetDict(data_dict)
    combined_path = os.path.join(config.logprob_cache_path, "teacher_logprobs")
    dataset.save_to_disk(combined_path)

    return combined_path


def cache_teacher_logprobs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = teacher_model.to(device)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    print(f"Teacher model (7B) loaded")

    dataset = get_dataset()
    train_dataloader, test_dataloader = prepare_dataset(dataset["train"], dataset["test"])
    breakpoint()

    print("\n--> Generating Teacher Logits")
    for split, dataloader in [("train", train_dataloader), ("test", test_dataloader)]:
        save_dir = os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}")
        os.makedirs(save_dir, exist_ok=True)

        save_ds = {"input_ids": [], "attention_mask": [], "labels": [], "logprob_values": [], "logprob_indices": [], "id": []}
        chunk_id = 0
        samples_in_chunk = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {split}")):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]  # Keep labels on CPU for now
                
                outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [batch, seq_len, vocab_size]
                
                logprobs = F.log_softmax(logits, dim=-1)
                values, indices = torch.topk(logprobs, k=100, dim=-1)
                values = values.to('cpu')
                indices = indices.to(torch.int32).to('cpu')

                for b in range(values.size(0)):
                    if not (labels[b] != -100).any():
                        continue

                    pad_positions = torch.where(batch["input_ids"][b] == tokenizer.pad_token_id)[0]
                    end_idx = pad_positions[0].item() if len(pad_positions) > 0 else len(batch["input_ids"][b])

                    save_ds["id"].append(batch["id"][b])
                    save_ds["input_ids"].append(batch["input_ids"][b].tolist())
                    save_ds["attention_mask"].append(batch["attention_mask"][b].tolist())
                    save_ds["labels"].append(batch["labels"][b].tolist())
                    save_ds["logprob_values"].append(values[b][:end_idx].tolist())
                    save_ds["logprob_indices"].append(indices[b][:end_idx].tolist())
                    
                    samples_in_chunk += 1

                # Save chunk periodically (every 3200 samples)
                if samples_in_chunk >= 3200:
                    print(f"--> [{split}] Saving chunk {chunk_id} with {len(save_ds['input_ids'])} samples")

                    save_path = os.path.join(save_dir, f"chunk_{chunk_id}.arrow")
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path)
                    Dataset.from_dict(save_ds).save_to_disk(save_path)

                    save_ds = {
                        "input_ids": [],
                        "attention_mask": [],
                        "labels": [],
                        "logprob_values": [],
                        "logprob_indices": [],
                        "id": [],
                    }
                    chunk_id += 1
                    samples_in_chunk = 0

        # Save any remaining data
        if save_ds["input_ids"]:
            print(f"--> [{split}] Saving final chunk {chunk_id} with {len(save_ds['input_ids'])} samples")
            save_path = os.path.join(save_dir, f"chunk_{chunk_id}.arrow")
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            Dataset.from_dict(save_ds).save_to_disk(save_path)
    
    print("\n--> Generation Done")
    print("\n--> Building Teacher Logits Dataset")
    build_teacher_logprobs_dataset()
    print("\n--> Building Done")


if __name__ == "__main__":
    cache_teacher_logprobs()
