import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import config
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

dataset = datasets.load_from_disk(config.dataset_path)
dataloader = DataLoader(dataset["train"].select(range(100)), batch_size=2)

ds = {"input_ids": [], "attention_mask": [], "labels": []}

teacher = AutoModelForCausalLM.from_pretrained(
    config.teacher_model_name,
    torch_dtype=torch.bfloat16,
    device_map=config.teacher_device,
)
student = AutoModelForCausalLM.from_pretrained(
    config.student_model_name,
    torch_dtype=torch.bfloat16,
    device_map=config.teacher_device,
)
teacher.resize_token_embeddings(new_num_tokens=student.vocab_size)
del student
teacher.eval()

print("\n=== GENERATING TEACHER LOGITS ===")
for batch in tqdm(dataloader):
    input_ids = batch["input_ids"].to(config.teacher_device)
    attention_mask = batch["attention_mask"].to(config.teacher_device)

    ds["input_ids"].append(input_ids.squeeze(0).cpu())
    ds["attention_mask"].append(attention_mask.squeeze(0).cpu())

    with torch.no_grad():
        generation_output = teacher.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=0.5,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_sequences = generation_output.sequences
        gen_only = generated_sequences[:, input_ids.shape[1] :]
        pad = torch.full((gen_only.size(0), input_ids.size(1)), fill_value=-100, dtype=gen_only.dtype, device=gen_only.device)
        labels = torch.cat([pad, gen_only], dim=1)

        ds["labels"].append(labels.squeeze(0).cpu())

        del generation_output, generated_sequences, gen_only, pad, input_ids, attention_mask
        torch.cuda.empty_cache()

dset = datasets.Dataset.from_dict(ds)
dset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dset.save_to_disk(config.synthetic_dataset_path)
