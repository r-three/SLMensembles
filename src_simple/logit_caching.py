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



    def get_teacher_logprobs(self):
        """Load or generate teacher logits dataset."""
        if not os.path.exists(os.path.join(config.logprob_cache_path, "teacher_logprobs")):
            self.cache_teacher_logprobs()

        main_print("--> Loading Teacher Logits")
        dataset = datasets.load_from_disk(os.path.join(config.logprob_cache_path, "teacher_logprobs"))
        
        return dataset

    def concatenate_logprobs_chunks(self, split_dirs: list[str]):
        """Concatenate logit chunks into a single dataset."""
        datasets_list = []
        for split_dir in split_dirs:
            chunk_paths = sorted(
                [p for p in glob.glob(os.path.join(split_dir, "chunk_*")) if os.path.isdir(p)],
                key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
            )
            main_print(f"--> Loading {len(chunk_paths)} chunks for '{os.path.basename(split_dir)}' split")
            datasets_list.extend(load_from_disk(p) for p in chunk_paths)
        return concatenate_datasets(datasets_list) if datasets_list else None

    def build_teacher_logprobs_dataset(self):
        """Build the final teacher logits dataset from chunks."""
        data_dict = {}

        for split in ["train", "test"]:
            split_dirs = sorted(
                [d for d in glob.glob(os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}_*")) if os.path.isdir(d)]
            )
            split_ds = self.concatenate_logprobs_chunks(split_dirs)
            data_dict[split] = split_ds

        dataset = DatasetDict(data_dict)
        combined_path = os.path.join(config.logprob_cache_path, "teacher_logprobs")
        dataset.save_to_disk(combined_path)

        return combined_path

def cache_teacher_logprobs(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    print("\n--> Generating Teacher Logits")
    for split in ["train", "test"]:
        save_dir = os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}")
        os.makedirs(save_dir, exist_ok=True)

        save_ds = {"input_ids": [], "attention_mask": [], "labels": [], "logprob_values": [], "logprob_indices": [], "start_idx": [], "end_idx": [], "id": []}
        chunk_id = 0

        batch_size = 6

        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            for batch in tqdm(f'{split}_dataloader', total=math.ceil(len(f'{split}_dataloader')/batch_size)):
                outputs = teacher_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits  # [batch, seq_len, vocab_size]
                breakpoint()
                # logprobs = F.log_softmax(logits, dim=-1)

                values, indices = torch.topk(logits, k=100, dim=-1)
                values = values.to('cpu')
                indices = indices.to(torch.int32).to('cpu')


                for b in range(values.size(0)):
                    start = torch.where(batch_data['labels'][b] != -100)[0] 
                    if len(start) == 0:
                        continue
                    start_idx = start[0].item()
                    
                    end = torch.where(batch_data['input_ids'][b] == tokenizer.pad_token_id)[0]
                    end_idx = end[0].item() if len(end) != 0 else len(batch_data['input_ids'][b]) - 1
                    
                    save_ds["id"].append(batch_data["id"][b])
                    save_ds["input_ids"].append(batch_data["input_ids"][b][:end_idx].tolist())
                    save_ds["attention_mask"].append(batch_data["attention_mask"][b][:end_idx].tolist())
                    save_ds["labels"].append(batch_data["labels"][b][:end_idx].tolist())
                    save_ds["logprob_values"].append(values[b][start_idx:end_idx].tolist())
                    save_ds["logprob_indices"].append(indices[b][start_idx:end_idx].tolist())
                    save_ds["start_idx"].append(start_idx)
                    save_ds["end_idx"].append(end_idx)

                batch_data = {"input_ids": [], "attention_mask": [], "labels": [], "id": []}

                if (batch_idx + 1) % 3200 < batch_size or batch_idx == len(shard) - 1:
                    main_print(f"--> [{split}] Saving chunk {chunk_id} with {len(save_ds['input_ids'])} samples")

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
                        "start_idx": [],
                        "end_idx": [],
                        "id": [],
                    }
                    chunk_id += 1

                # if (batch_idx + 1) % 3200 == 0:
                #     break

        if save_ds["input_ids"]:
            print(f"--> [{split}] Saving final chunk {chunk_id} with {len(save_ds['input_ids'])} samples")
            save_path = os.path.join(save_dir, f"chunk_{chunk_id}.arrow")
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            Dataset.from_dict(save_ds).save_to_disk(save_path)
    
    self.build_teacher_logprobs_dataset()
    
    main_print("\n--> Generation Done")




if __name__ == "__main__":
    dataClass = DistillDataset()
    dataset = dataClass.get_dataset()
    teacher_logits = dataClass.cache_teacher_logprobs()
