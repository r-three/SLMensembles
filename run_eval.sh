#!/bin/bash
# Example script for running evaluation

# Example 1: Evaluate a local checkpoint
# python src_simple/simple_eval.py \
#     --model_path /scratch/klambert/model_log/singular/checkpoints/checkpoint_epoch0_step5000.pt \
#     --device cuda

# Example 2: Evaluate a HuggingFace model
# python src_simple/simple_eval.py \
#     --model_name allenai/OLMo-2-0425-1B-SFT \
#     --device cuda

# Example 3: Evaluate the teacher model
# python src_simple/simple_eval.py \
#     --model_name allenai/OLMo-2-1124-7B-SFT \
#     --device cuda

# Uncomment one of the examples above or pass your own arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [--model_path PATH | --model_name NAME] [--device DEVICE]"
    echo ""
    echo "Examples:"
    echo "  $0 --model_path /scratch/klambert/model_log/singular/checkpoints/checkpoint_epoch0_step5000.pt"
    echo "  $0 --model_name allenai/OLMo-2-0425-1B-SFT"
    exit 1
fi

cd /home/klambert/projects/aip-craffel/klambert/SLMensembles
python src_simple/simple_eval.py "$@"

