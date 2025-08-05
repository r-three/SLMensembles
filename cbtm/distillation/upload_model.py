#!/usr/bin/env python3
"""
Push Fine-tuned Qwen Model to Hugging Face Hub
Uploads your distilled/fine-tuned model with proper documentation
"""

import os
import argparse
import sys
from pathlib import Path
import json
import shutil
from datetime import datetime

try:
    from huggingface_hub import HfApi, login, Repository, create_repo
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("📦 Install with: pip install huggingface_hub transformers torch")
    sys.exit(1)

def authenticate_hf():
    """Authenticate with Hugging Face Hub"""
    print("🔐 Authenticating with Hugging Face Hub...")
    
    # Try to login (will use cached token if available)
    try:
        login()
        print("✅ Successfully authenticated!")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\n🔧 To fix this:")
        print("1. Get your token from: https://huggingface.co/settings/tokens")
        print("2. Run: huggingface-cli login")
        print("3. Or set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token")
        return False

def validate_model_path(model_path):
    """Validate that the model path contains necessary files"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    required_files = [
        "config.json",
        "pytorch_model.bin",  # or model.safetensors
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            # Check for alternative file names
            if file == "pytorch_model.bin":
                if not any((model_path / alt).exists() for alt in ["model.safetensors", "pytorch_model-00001-of-00001.bin"]):
                    missing_files.append(file)
            else:
                missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files in model directory: {missing_files}")
        print("🔍 Available files:")
        for file in model_path.iterdir():
            print(f"  • {file.name}")
    
    return len(missing_files) == 0

def test_model_loading(model_path):
    """Test that the model can be loaded properly"""
    print("🧪 Testing model loading...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        print("✅ Model loads successfully!")
        
        # Test a simple generation
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Test generation works: '{response[:50]}...'")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def push_to_hub(model_path, repo_name, private=False, commit_message=None):
    """Push the model to Hugging Face Hub"""
    
    model_path = Path(model_path)
    
    # Validate inputs
    if not validate_model_path(model_path):
        print("❌ Model validation failed!")
        return False
    
    # Test model loading
    if not test_model_loading(model_path):
        print("❌ Model testing failed!")
        return False
    
    # Authenticate
    if not authenticate_hf():
        return False
    
    try:
        api = HfApi()
        
        # Create repository
        print(f"📁 Creating repository: {repo_name}")
        create_repo(
            repo_id=repo_name,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print("✅ Repository created/verified!")    

        
        # Upload all files
        print("📤 Uploading files to Hub...")
        
        commit_msg = commit_message or f"Upload fine-tuned Qwen2.5-1.5B model - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_msg,
            ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store"]
        )
        
        print("✅ Upload completed successfully!")
        print(f"🌐 Model available at: https://huggingface.co/{repo_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push fine-tuned Qwen model to Hugging Face Hub")
    parser.add_argument("--model_path", "-m", 
                       default="/home/ehghaghi/scratch/ehghaghi/distillation_results/8_coding/checkpoint-421",
                       help="Path to fine-tuned model directory")
    parser.add_argument("--repo_name", "-r", required=True,
                       help="Repository name on HF Hub (e.g., 'username/qwen2.5-1.5b-distilled')")
    parser.add_argument("--private", "-p", action="store_true",
                       help="Make repository private")
    parser.add_argument("--commit_message", "-c",
                       help="Custom commit message")
    parser.add_argument("--test_only", "-t", action="store_true",
                       help="Only test model loading, don't upload")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    print("🚀 Hugging Face Model Upload Tool")
    print(f"📂 Model path: {args.model_path}")
    print(f"📁 Repository: {args.repo_name}")
    print(f"🔒 Private: {args.private}")
    
    # Validation
    if not Path(args.model_path).exists():
        print(f"❌ Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Test mode
    if args.test_only:
        print("\n🧪 Test mode - checking model loading only")
        if test_model_loading(args.model_path):
            print("✅ Model is ready for upload!")
        else:
            print("❌ Model has issues!")
        return
    
    # Confirmation
    if not args.force:
        print(f"\n⚠️  About to upload model to: https://huggingface.co/{args.repo_name}")
        print(f"📁 Source: {args.model_path}")
        print(f"🔒 Private: {args.private}")
        
        confirm = input("\n❓ Continue? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Upload cancelled!")
            return
    
    # Upload
    success = push_to_hub(
        args.model_path,
        args.repo_name,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if success:
        print("\n🎉 Success! Your model is now available on Hugging Face Hub!")
        print(f"🔗 URL: https://huggingface.co/{args.repo_name}")
        print(f"💻 Load with: AutoModelForCausalLM.from_pretrained('{args.repo_name}')")
    else:
        print("\n❌ Upload failed! Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()