import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import yaml



# Configuration
config = {
    "project_name": "SLMensembles",
    "dataset": {
        "name": "Malikeh1375/clustered_tulu_3_8",
        "config_name": "programming_and_code_development",
        "split": "train",
        "num_samples": 25000, # You can pass a number here to limit the number of samples to use.
        "seed": 1997
    },
    "models": {
        "teacher": "Qwen/Qwen2.5-7B-Instruct",
        "student": "Qwen/Qwen2.5-1.5B-Instruct"
    },
    "tokenizer": {
        "max_length": 1024,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 500,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True,
        # ADD THESE EVALUATION PARAMETERS
       # ADD THESE EVALUATION PARAMETERS
        "eval_strategy": "steps",  # or "epoch" (updated parameter name)
        "eval_steps": 10,
        "per_device_eval_batch_size": 1,
        "dataloader_num_workers": 2, 
        # Remove include_for_metrics entirely for now to avoid compatibility issues
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "max_grad_norm": 1
    },
    "distillation": {
        "temperature": 1.0,
        "alpha": 0.5,
        "distillation_type": "forward_kld"  # Added distillation type
    },
    "model_config": {
        "use_flash_attention": False
    }
    # "spectrum": {
    #     "layers_to_unfreeze": "/workspace/spectrum/snr_results_Qwen-Qwen2-1.5B_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
    # }
}

# Set up environment
os.environ['WANDB_PROJECT'] = config["project_name"]
config["training"]["output_dir"] = os.environ['OUTPUT_DIR']

# Load and preprocess dataset
dataset = load_dataset(config["dataset"]["name"], name=config["dataset"]["config_name"], split=config["dataset"]["split"])
dataset = dataset.shuffle(seed=config["dataset"]["seed"])
if "num_samples" in config["dataset"]:
    dataset = dataset.select(range(config["dataset"]["num_samples"]))

print(f"Dataset size after selection: {len(dataset)}")

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

# Apply chat template to student tokenizer
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

# ----------------------------------
# FIXED DIAGNOSTIC AND HELPER FUNCTIONS
# ----------------------------------
def create_response_labels_fixed(input_ids, tokenizer, attention_mask):
    """Fixed label creation that excludes padding tokens"""
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
        
    labels = input_ids.clone()
    response_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    labels.fill_(-100)  # Mask all tokens initially
    
    # Find where assistant response starts
    start_pos = -1
    for i in range(len(input_ids) - len(response_ids) + 1):
        if input_ids[i : i + len(response_ids)].tolist() == response_ids:
            start_pos = i + len(response_ids)
            break
    
    if start_pos == -1:
        return labels  # No assistant response found
    
    # Find where real content ends (before padding)
    real_length = len(input_ids)
    
    # Find the last non-padding token
    for i in range(len(attention_mask) - 1, -1, -1):
        if attention_mask[i] == 1:
            real_length = i + 1
            break
    
    # Only unmask assistant response tokens that are not padding
    end_pos = min(len(input_ids), real_length)
    if start_pos < end_pos:
        labels[start_pos:end_pos] = input_ids[start_pos:end_pos]
    
    return labels

def tokenize_and_label_fixed(example):
    """Fixed tokenization that preserves assistant responses and excludes padding"""
    # First, tokenize WITHOUT padding to see the actual length
    tokenized_no_pad = student_tokenizer(
        example["chat_text"],
        truncation=False,  # Don't truncate yet
        padding=False,     # Don't pad yet
        return_tensors="pt",
    )
    
    actual_length = len(tokenized_no_pad["input_ids"][0])
    max_length = config["tokenizer"]["max_length"]
    
    # If too long, we need to truncate intelligently
    if actual_length > max_length:
        # Find the assistant response start
        text = example["chat_text"]
        assistant_start = text.find("<|im_start|>assistant\n")
        
        if assistant_start == -1:
            # No assistant response found, skip this example
            return None
            
        # Tokenize just the prefix to see how much space we need
        prefix = text[:assistant_start + len("<|im_start|>assistant\n")]
        prefix_tokens = student_tokenizer.encode(prefix)
        
        # Reserve space for the assistant response (at least 50 tokens)
        min_response_tokens = 50
        max_prefix_tokens = max_length - min_response_tokens
        
        if len(prefix_tokens) > max_prefix_tokens:
            # Prefix is too long, truncate it
            prefix_tokens = prefix_tokens[:max_prefix_tokens]
            prefix = student_tokenizer.decode(prefix_tokens, skip_special_tokens=False)
            
            # Reconstruct the text with truncated prefix
            assistant_text = text[assistant_start + len("<|im_start|>assistant\n"):]
            remaining_tokens = max_length - len(prefix_tokens)
            
            # Tokenize assistant response and keep as much as possible
            assistant_tokens = student_tokenizer.encode(assistant_text)[:remaining_tokens-1]  # -1 for safety
            assistant_text = student_tokenizer.decode(assistant_tokens, skip_special_tokens=False)
            
            # Reconstruct full text
            text = prefix + assistant_text
        else:
            # Truncate from the end to preserve assistant response start
            text = text[:max_length * 4]  # Rough estimate, will be refined below
    else:
        text = example["chat_text"]
    
    # Now tokenize with proper settings
    tokenized = student_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    
    input_ids = tokenized["input_ids"].squeeze(0)
    attention_mask = tokenized["attention_mask"].squeeze(0)
    
    # Create labels - FIXED to not include padding tokens
    labels = create_response_labels_fixed(input_ids, student_tokenizer, attention_mask)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def filter_valid_examples_improved(example):
    """Improved filtering for highly variable response lengths"""
    if example is None:
        return False
        
    labels = example['labels']
    unmasked_count = (labels != -100).sum().item()
    total_tokens = len(labels)
    unmasked_ratio = unmasked_count / total_tokens
    
    # Filter out extremely short responses (like "Ọgụ")
    if unmasked_count < 5:  # Less than 5 tokens is too short
        return False
    
    # Filter out extremely long responses that dominate the sequence
    if unmasked_count > 1000:  # More than 1000 tokens is too long
        return False
    
    # More nuanced ratio filtering
    if unmasked_ratio < 0.01:  # Less than 1% is too short
        return False
        
    if unmasked_ratio > 0.60:  # More than 60% means input is too short
        return False
    
    # Additional check: ensure we have reasonable input length too
    masked_count = (labels == -100).sum().item()
    if masked_count < 20:  # Input (system + user) should be at least 20 tokens
        return False
    
    return True


def analyze_original_dataset(dataset, num_samples=5):
    """Analyze the original dataset structure before any processing"""
    print("=== ORIGINAL DATASET ANALYSIS ===")
    print(f"Dataset columns: {dataset.column_names}")
    
    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Available keys: {list(example.keys())}")
        
        # Analyze the messages structure
        if 'messages' in example:
            messages = example['messages']
            print(f"  Messages type: {type(messages)}")
            print(f"  Number of messages: {len(messages) if isinstance(messages, list) else 'Not a list'}")
            
            if isinstance(messages, list):
                assistant_count = 0
                user_count = 0
                
                for j, msg in enumerate(messages):
                    if isinstance(msg, dict):
                        role = msg.get('role', 'unknown')
                        # Try both 'value' and 'content' fields
                        content = msg.get('value', msg.get('content', ''))
                        
                        if role == 'assistant':
                            assistant_count += 1
                            print(f"    Message {j}: ASSISTANT (length: {len(content)})")
                            if len(content) > 0:
                                print(f"      Preview: '{content[:100]}...'")
                            else:
                                print(f"      ⚠️  EMPTY ASSISTANT CONTENT!")
                        elif role == 'user':
                            user_count += 1
                            print(f"    Message {j}: USER (length: {len(content)})")
                        else:
                            print(f"    Message {j}: ROLE '{role}' (length: {len(content)})")
                
                print(f"  Role counts - User: {user_count}, Assistant: {assistant_count}")
                
                if assistant_count == 0:
                    print(f"  ⚠️  WARNING: No assistant messages found in sample {i}!")

def analyze_formatted_data(dataset, num_samples=5):
    """Analyze data after chat template is applied"""
    print("=== FORMATTED DATA ANALYSIS ===")
    
    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        text = example.get('chat_text', '')
        
        print(f"\nSample {i}:")
        print(f"  Chat text length: {len(text)} characters")
        
        # Show the structure
        print(f"  Full text preview (first 300 chars):")
        print(f"    '{text[:300]}...'")
        
        # Count different sections
        system_count = text.count("<|im_start|>system\n")
        user_count = text.count("<|im_start|>user\n") 
        assistant_count = text.count("<|im_start|>assistant\n")
        
        print(f"  Section counts - System: {system_count}, User: {user_count}, Assistant: {assistant_count}")
        
        # Find assistant response
        assistant_start = text.find("<|im_start|>assistant\n")
        if assistant_start != -1:
            assistant_text = text[assistant_start + len("<|im_start|>assistant\n"):]
            assistant_end = assistant_text.find("<|im_end|>")
            if assistant_end != -1:
                actual_response = assistant_text[:assistant_end]
                print(f"  Assistant response length: {len(actual_response)} characters")
                print(f"  Assistant response preview: '{actual_response[:100]}...'")
            else:
                print("  ⚠️  WARNING: No <|im_end|> found after assistant start!")
                # Check if it ends the string
                if assistant_text.strip():
                    print(f"  Assistant text without end marker: '{assistant_text[:100]}...'")
        else:
            print("  ⚠️  WARNING: No assistant response found!")

def fixed_sharegpt_format(example):
    """FIXED: Format messages using chat template"""
    conversations = example['messages']
    message = []
    
    assistant_found = False
    user_found = False
    
    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict):
                role = conversation.get('role', '')
                # Try both 'value' and 'content' fields - FIXED
                content = conversation.get('value', conversation.get('content', ''))
                
                if role == 'user':
                    user_found = True
                    message.append({"role": "user", "content": content})
                elif role == 'assistant':
                    assistant_found = True
                    message.append({"role": "assistant", "content": content})
                elif role == 'system':
                    message.insert(0, {"role": "system", "content": content})

    # Add default system message if none exists
    if not any(msg.get('role') == 'system' for msg in message):
        message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    # Warning if no assistant message found
    if not assistant_found:
        print(f"⚠️  WARNING: No assistant message found in this example!")
        return {"chat_text": ""}  # Return empty to filter out later
    
    if not user_found:
        print(f"⚠️  WARNING: No user message found in this example!")

    try:
        # FIXED: Use add_generation_prompt=False since we have assistant responses
        text = student_tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=False  # CRITICAL FIX!
        )
        return {"chat_text": text}
    except Exception as e:
        print(f"ERROR applying chat template: {e}")
        print(f"Messages: {message}")
        return {"chat_text": ""}

def analyze_dataset_labels(dataset, tokenizer, num_samples=20):
    """Analyze label distribution and response lengths"""
    print("=== DATASET LABEL ANALYSIS ===")
    
    total_unmasked = 0
    total_tokens = 0
    response_lengths = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        # Count tokens
        seq_length = len(input_ids)
        masked_tokens = (labels == -100).sum().item()
        unmasked_tokens = seq_length - masked_tokens
        
        total_unmasked += unmasked_tokens
        total_tokens += seq_length
        response_lengths.append(unmasked_tokens)
        
        if i < 5:  # Show details for first 5
            print(f"\nSample {i}:")
            print(f"  Sequence length: {seq_length}")
            print(f"  Unmasked tokens: {unmasked_tokens}")
            print(f"  Unmasked ratio: {unmasked_tokens/seq_length:.1%}")
            
            # Show the actual assistant response
            unmasked_indices = (labels != -100).nonzero().flatten()
            if len(unmasked_indices) > 0:
                response_text = tokenizer.decode(input_ids[unmasked_indices])
                print(f"  Response: '{response_text[:100]}...'")
# ----------------------------------
# FIXED DATA PROCESSING PIPELINE
# ----------------------------------

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names

# STEP 0: Analyze the ORIGINAL dataset structure - FIXED
print("=== ANALYZING ORIGINAL DATASET ===")
analyze_original_dataset(dataset, num_samples=3)

# Step 1: Format chat data with FIXED function
print("\n=== APPLYING CHAT TEMPLATE ===")
dataset = dataset.map(fixed_sharegpt_format, remove_columns=original_columns)

# Filter out empty examples early
print("=== FILTERING EMPTY CHAT TEXTS ===")
original_size = len(dataset)
dataset = dataset.filter(lambda x: len(x.get('chat_text', '')) > 0)
print(f"Filtered out {original_size - len(dataset)} empty examples")

if len(dataset) == 0:
    print("ERROR: All examples were filtered out! Check your data format.")
    exit(1)

# STEP 1.5: Analyze formatted data - FIXED function call
print("=== ANALYZING FORMATTED DATA ===")
analyze_formatted_data(dataset, num_samples=3)

# Continue with tokenization...
print("=== TOKENIZING AND CREATING LABELS (FIXED) ===")
tokenized_dataset = dataset.map(
    tokenize_and_label_fixed, 
    remove_columns=["chat_text"], 
    num_proc=8
).filter(lambda x: x is not None)  # Remove failed examples

# Add analysis of tokenized data
print("=== ANALYZING TOKENIZED DATA ===")
if len(tokenized_dataset) > 0:
    # Set format temporarily for analysis
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    analyze_dataset_labels(tokenized_dataset, student_tokenizer, num_samples=5)
    
    # Filter valid examples
    print("=== FILTERING VALID EXAMPLES ===")
    print(f"Dataset size before filtering: {len(tokenized_dataset)}")
    
    filtered_dataset = tokenized_dataset.filter(filter_valid_examples_improved, num_proc=8)
    print(f"Dataset size after filtering: {len(filtered_dataset)}")
    
    if len(filtered_dataset) == 0:
        print("ERROR: No valid examples remain after filtering!")
        print("This suggests the assistant responses are too short or malformed.")
        
        # Debug: show what's happening with labels
        print("\n=== DEBUGGING LABEL CREATION ===")
        sample = tokenized_dataset[0]
        labels = sample['labels']
        input_ids = sample['input_ids']
        
        unmasked_count = (labels != -100).sum().item()
        print(f"Sample unmasked tokens: {unmasked_count}")
        print(f"Total tokens: {len(labels)}")
        
        if unmasked_count > 0:
            unmasked_indices = (labels != -100).nonzero().flatten()
            response_text = student_tokenizer.decode(input_ids[unmasked_indices])
            print(f"Actual response text: '{response_text}'")
        
        exit(1)
    
    # Train/test split
    tokenized_dataset = filtered_dataset.train_test_split(test_size=0.1)
    print(f"Final train size: {len(tokenized_dataset['train'])}")
    print(f"Final test size: {len(tokenized_dataset['test'])}")
else:
    print("ERROR: No examples survived tokenization!")
    exit(1)


print("Dataset preparation complete. Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)

# Optionally freeze layers of the student model based on spectrum configuration
if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
    def freeze_student_spectrum(model, unfrozen_layers_file):
        with open(unfrozen_layers_file, 'r') as file:
            unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
        
        for name, param in model.named_parameters():
            if not any(layer in name for layer in unfrozen_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Apply freezing to student model
    freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"])
else:
    print("Spectrum configuration not found. All layers of the student model will be trainable.")

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits

class LogitsTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_evaluating = False
        self.eval_ce_losses = []
        self.eval_kl_losses = []
        
        # Simplified logging - removed complex accumulation
        import time
        self.training_start_time = time.time()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Move teacher model to device only once if not already there
        if self.teacher_model.device != device:
            self.teacher_model = self.teacher_model.to(device)
        
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss, ce_loss, kl_loss = self.distillation_loss(model, student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss)
        
        # Simplified logging
        if not self.is_evaluating and hasattr(self, 'log') and self.state.global_step % self.args.logging_steps == 0:
            import time
            current_time = time.time()
            total_elapsed = current_time - self.training_start_time
            
            self.log({
                "train/cross_entropy_loss": ce_loss.item(),
                "train/kl_divergence_loss": kl_loss.item(),
                "train/combined_loss": custom_loss.item(),
                "train/total_elapsed_hours": total_elapsed / 3600,
                "train/steps_per_hour": self.state.global_step / (total_elapsed / 3600) if total_elapsed > 0 else 0,
            })
        elif self.is_evaluating:
            # Store losses for evaluation averaging
            self.eval_ce_losses.append(ce_loss.item())
            self.eval_kl_losses.append(kl_loss.item())
        
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, model, student_logits, teacher_logits, inputs, original_loss):
        device = next(model.parameters()).device
        student_logits, teacher_logits = pad_logits(student_logits.to(device), teacher_logits.to(device))
        
        temperature = config["distillation"]["temperature"]
        alpha = config["distillation"]["alpha"]

        labels = inputs.get('labels', None)
        
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_student_logits = student_logits[..., :-1, :].contiguous()
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Mask out -100 positions
            valid_mask = (shift_labels != -100).float()
            
            if valid_mask.sum() > 0:
                # Scale logits
                student_scaled = F.log_softmax(shift_student_logits / temperature, dim=-1)
                teacher_scaled = F.softmax(shift_teacher_logits / temperature, dim=-1)
                
                # Compute KL divergence
                loss_kd = F.kl_div(
                    student_scaled,
                    teacher_scaled,
                    reduction='none'
                ).sum(dim=-1)  # Sum over vocab

                # Mask and normalize
                loss_kd = (loss_kd * valid_mask).sum() / valid_mask.sum()
                loss_kd = loss_kd * (temperature ** 2)
            else:
                loss_kd = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # No labels provided – fallback to batchmean over all tokens
            student_scaled = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_scaled = F.softmax(teacher_logits / temperature, dim=-1)

            loss_kd = F.kl_div(
                student_scaled,
                teacher_scaled,
                reduction='batchmean'
            ) * (temperature ** 2)

        ce_loss = original_loss
        total_loss = (1 - alpha) * loss_kd + alpha * ce_loss

        return total_loss, ce_loss, loss_kd

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluation loop to set evaluation flag and log metrics"""
        self.is_evaluating = True
        self.eval_ce_losses = []
        self.eval_kl_losses = []
        
        result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        # Log averaged evaluation metrics
        if self.eval_ce_losses and self.eval_kl_losses:
            avg_ce_loss = sum(self.eval_ce_losses) / len(self.eval_ce_losses)
            avg_kl_loss = sum(self.eval_kl_losses) / len(self.eval_kl_losses)
                
            # Update the result metrics dictionary
            result.metrics.update({
                "eval_cross_entropy_loss": avg_ce_loss,
                "eval_kl_divergence_loss": avg_kl_loss,
                "eval_combined_loss": (1-config["distillation"]["alpha"]) * avg_kl_loss + config["distillation"]["alpha"] * avg_ce_loss,
            })
        
        self.is_evaluating = False
        return result

# Training arguments
training_arguments = TrainingArguments(**config["training"])

# Create the custom SFT Trainer
trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_arguments,
)

# Add the teacher model to the trainer
trainer.teacher_model = teacher_model

# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])