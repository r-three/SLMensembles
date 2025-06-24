import psutil
import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import config
import torch.nn.functional as F

dataset = datasets.load_from_disk(config.dataset_path)

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
ds = {"input_ids": [], "attention_mask": [], "labels": []}

for batch in tqdm(dataset["train"]):
    input_ids = batch["input_ids"].to(config.teacher_device)
    attention_mask = batch["attention_mask"].to(config.teacher_device)

    with torch.no_grad():
        generation_output = teacher.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=0.5,
            max_length=1024,
        )
        text = tokenizer.decode(generation_output.sequences[0])
        ds["text"].append(text)

        generated_sequences = generation_output.sequences
        gen_only = generated_sequences[:, input_ids.shape[1] :]

        pad = torch.full((gen_only.size(0), input_ids.size(1)), fill_value=-100, dtype=gen_only.dtype)
        labels = torch.cat([pad, gen_only], dim=1)

        # Clean up GPU memory
        del generation_output, generated_sequences, gen_only, pad, input_ids, attention_mask
        torch.cuda.empty_cache()

        batch["labels"] = labels

dset = datasets.Dataset.from_dict(ds)  # turn into huggingface dict
dset.save_to_disk(config.synthetic_dataset_path)
dset.set_format(type="torch", columns=["text"])

print("\n=== GENERATING TEACHER LOGITS ===")
tokenized_final = dataset.map(
    add_teacher_logits,
    batched=True,
    writer_batch_size=10,
    batch_size=10,
    with_rank=True,
    num_proc=torch.cuda.device_count(),
    load_from_cache_file=False,
)
tokenized_final.save_to_disk(config.synthetic_dataset_path)
