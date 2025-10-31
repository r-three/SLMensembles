import argparse
import sys
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from simple_utils import prepare_dataset, get_dataset, fix_seed
from simple_config import config

def load_model(model_path=None, model_name=None, device='cuda'):
    """Load model from either local checkpoint or HuggingFace."""
    if model_path: 
        # Load from local checkpoint
        print(f"Loading from checkpoint: {model_path}") 
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
    elif model_name:
        # Load from HuggingFace
        print(f"Loading from HuggingFace: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    
    return model.to(device)
    

def compute_ce_loss(model, dataloader, device):
    """Compute loss on the test dataset."""
    model.eval()
    
    total_ce_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    print("\nEvaluating model...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", file=sys.stdout)):
            input_ids = batch["input_ids"].type(torch.LongTensor).to(device)
            attention_mask = batch["attention_mask"].type(torch.LongTensor).to(device)
            labels = batch["labels"].type(torch.LongTensor).to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for next-token prediction
            vocab_size = logits.size(-1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Create mask for valid tokens (ignore -100)
            ignore_index = -100
            valid_mask = shift_labels != ignore_index
            valid_count = valid_mask.sum().item()
            
            if valid_count > 0:
                # Compute cross-entropy loss
                ce_loss = F.cross_entropy(
                    shift_logits,
                    shift_labels,
                    ignore_index=ignore_index,
                    reduction='sum'
                )
                
                total_ce_loss += ce_loss.item()
                total_tokens += valid_count
                num_batches += 1
            
            # Periodic cleanup
            del outputs, logits, shift_logits, shift_labels
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
    
    # Compute averages
    if total_tokens > 0:
        avg_ce_loss = total_ce_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()
    else:
        avg_ce_loss = float('inf')
        perplexity = float('inf')
    
    return avg_ce_loss, perplexity, num_batches


def eval_main(args):
    """Main evaluation function."""
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Load dataset
    print("Loading test dataset...")
    dataset = get_dataset()
    
    # Create dataloader
    _, eval_dataloader = prepare_dataset(
        dataset['train'],
        dataset['test'],
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        model_path=args.model_path,
        model_name=args.model_name,
        device=device
    )
    
    # Evaluate
    avg_ce_loss, perplexity, num_batches = compute_ce_loss(
        model, eval_dataloader, device
    )
    
    # Print results
    if args.model_path:
        print(f"Model: {args.model_path}")
    else:
        print(f"Model: {args.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Test examples: {len(dataset['test'])}")
    print(f"Batches processed: {num_batches}")
    print(f"Cross-Entropy Loss: {avg_ce_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("="*70)
    
    return avg_ce_loss, perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on the test dataset")
    
    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_path", type=str, help="Path to local checkpoint (e.g., /scratch/klambert/model_log/singular/checkpoints/checkpoint_epoch0_step5000.pt)")
    model_group.add_argument("--model_name", type=str, help="HuggingFace model name (e.g., allenai/OLMo-2-0425-1B-SFT)")
    
    args = parser.parse_args()
    eval_main(args)



