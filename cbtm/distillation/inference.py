import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse

def load_model(model_path, device="auto"):
    """Load the fine-tuned Qwen model and tokenizer"""
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Ensure proper padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print(f"✅ Model loaded successfully!")
    return model, tokenizer

def test_model(model, tokenizer, user_input):
    """Test the model with a single input"""
    
    # Format as chat messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"📝 Input: {user_input}")
    print("🔄 Generating response...")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (remove input tokens)
    input_length = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    generation_time = time.time() - start_time
    print(f"⏱️  Generated in {generation_time:.2f} seconds")
    print(f"📊 Generated {len(response_tokens)} tokens")
    print(f"\n🤖 Response:\n{response}")
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Simple model test")
    parser.add_argument("--model_path", "-m", 
                       default="/home/ehghaghi/scratch/ehghaghi/distillation_results/8_coding/checkpoint-421", 
                       help="Path to fine-tuned model")
    parser.add_argument("--input", "-i", 
                       default="Write a Python function to calculate fibonacci numbers recursively.",
                       help="Test input text")
    parser.add_argument("--device", "-d", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    print("🚀 Simple Model Test")
    print("=" * 50)
    
    try:
        # Load model
        model, tokenizer = load_model(args.model_path, args.device)
        
        # Test with input
        test_model(model, tokenizer, args.input)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()