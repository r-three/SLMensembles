#!/bin/bash

# Simple script to upload model to Hugging Face
# Usage: ./upload.sh

# Configuration - UPDATE THESE VALUES
MODEL_PATH="/home/ehghaghi/scratch/ehghaghi/distillation_results/8_non_english_mathematics/checkpoint-421"
HF_USERNAME="Malikeh1375"  # Change this to your HF username
REPO_NAME="Qwen2.5-1.5B-Non-English-Mathematics-Distilled-8Clusters"

echo "🚀 Uploading model to Hugging Face..."
echo "📂 Model: $MODEL_PATH"
echo "📁 Repo: $HF_USERNAME/$REPO_NAME"
echo ""

# Run the Python upload script
python3 upload_model.py \
  --model_path "$MODEL_PATH" \
  --repo_name "$HF_USERNAME/$REPO_NAME" \
  --commit_message "Upload fine-tuned Qwen2.5-1.5B non-english mathematics model"

echo ""
echo "✅ Done! Check: https://huggingface.co/$HF_USERNAME/$REPO_NAME"