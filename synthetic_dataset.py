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
batch_num = 0



def add_teacher_labels(example):
    global batch_num
    batch_num += 2
    
    with torch.no_grad():
        







def add_teacher_logits(batch):
    global batch_num

    batch_num += 2
    input_ids = batch["input_ids"].to(config.teacher_device)
    attention_mask = batch["attention_mask"].to(config.teacher_device)

    import pdb

    breakpoint()

    with torch.no_grad():
        generation_output = teacher.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=0.5,
            max_new_tokens=60,
        )
        print(f"Memory: {psutil.Process().memory_info().rss / (1024 * 1024)}, batch: {batch_num}")
        generated_sequences = generation_output.sequences
        gen_only = generated_sequences[:, input_ids.shape[1] :]

        pad = torch.full((gen_only.size(0), input_ids.size(1)), fill_value=-100, dtype=gen_only.dtype)
        labels = torch.cat([pad, gen_only], dim=1)

        batch["labels"] = labels
        return batch


print("\n=== GENERATING TEACHER LOGITS ===")
tokenized_final = dataset.map(add_teacher_logits, batched=True, batch_size=2, num_proc=1, load_from_cache_file=False)
tokenized_final.save_to_disk(config.synthetic_dataset_path)
