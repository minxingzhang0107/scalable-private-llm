#!/bin/bash

set -e

# =============================================================================
# Simple Author-Level DP LoRA Training
# =============================================================================

echo "🚀 Simple Author-Level DP LoRA Training Started: $(date)"
echo "✅ This script runs a high-epsilon (minimal privacy) training job."

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HOME/.cache/huggingface"

# Clear GPU memory before starting
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('🧹 GPU memory cleared')
"

# =============================================================================
# MODEL AND DATA CONFIGURATION
# =============================================================================

MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
DATA_FILE="dataset/private/tofu/tofu_train.json"
OUTPUT_DIR="./model_checkpoints/simple_lora_finetune_author_level_DPSGD_mistral_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# DP CONFIGURATION (High Epsilon = Minimal Privacy)
# =============================================================================

EPSILON=20.0
DELTA=1e-6
MAX_GRAD_NORM=8.0               # Standard clipping norm
USER_SAMPLING_RATE=0.1

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

EPOCHS=10
LEARNING_RATE=1e-5              # Standard learning rate for stable bfloat16 training
MAX_LENGTH=256
PER_AUTHOR_BATCH_SIZE=20

# =============================================================================
# LORA CONFIGURATION
# =============================================================================

LORA_R=8
LORA_ALPHA=8

# =============================================================================
# PRE-RUN CHECKS
# =============================================================================

# Assumes the python script is named based on its content/purpose
PYTHON_SCRIPT="src/lora_finetune/author_level_DPSGD_finetune_lora.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Missing Python script: $PYTHON_SCRIPT."
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Dataset file not found at $DATA_FILE."
    exit 1
fi

mkdir -p logs

# Check system
echo "🧮 System check..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

echo "✅ Pre-run checks completed."

# =============================================================================
# BUILD TRAINING COMMAND
# =============================================================================

CMD_ARGS="--model-name $MODEL_NAME"
CMD_ARGS="$CMD_ARGS --data-file $DATA_FILE"
CMD_ARGS="$CMD_ARGS --output-dir $OUTPUT_DIR"
CMD_ARGS="$CMD_ARGS --epsilon $EPSILON"
CMD_ARGS="$CMD_ARGS --delta $DELTA"
CMD_ARGS="$CMD_ARGS --max-grad-norm $MAX_GRAD_NORM"
CMD_ARGS="$CMD_ARGS --user-sampling-rate $USER_SAMPLING_RATE"
CMD_ARGS="$CMD_ARGS --epochs $EPOCHS"
CMD_ARGS="$CMD_ARGS --learning-rate $LEARNING_RATE"
CMD_ARGS="$CMD_ARGS --max-length $MAX_LENGTH"
CMD_ARGS="$CMD_ARGS --per-author-batch-size $PER_AUTHOR_BATCH_SIZE"
CMD_ARGS="$CMD_ARGS --lora-r $LORA_R"
CMD_ARGS="$CMD_ARGS --lora-alpha $LORA_ALPHA"

echo "✅ DP Configuration Summary:"
echo "   Model: $MODEL_NAME"
echo "   Data File: $DATA_FILE"
echo "   Privacy: ε=$EPSILON, δ=$DELTA, C=$MAX_GRAD_NORM"

SAMPLING_PERCENTAGE=$(python -c "print(f'{$USER_SAMPLING_RATE * 100:.1f}')")
echo "   Author Sampling: $USER_SAMPLING_RATE ($SAMPLING_PERCENTAGE%)"
echo "   📦 Examples per Author: $PER_AUTHOR_BATCH_SIZE"

echo "   Training: $EPOCHS epochs, LR=$LEARNING_RATE"
echo "   LoRA: r=$LORA_R, α=$LORA_ALPHA"
echo "   Max Length: $MAX_LENGTH tokens"

# =============================================================================
# RUN DP TRAINING
# =============================================================================

echo ""
echo "🚀 Starting Simple Author-Level DPSGD Training..."
echo "Command: python -u $PYTHON_SCRIPT $CMD_ARGS"
echo "============================================================"

start_time=$(date +%s)

# Execute the command and log output to a file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python -u "$PYTHON_SCRIPT" $CMD_ARGS 2>&1 | tee -a "logs/simple_lora_finetune_author_level_DPSGD_output_${TIMESTAMP}.log"
exit_code=${PIPESTATUS[0]}

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "============================================================"
echo "📅 Training completed at: $(date)"
echo "⏱️ Duration: ${duration}s ($(python -c "print(f'{$duration/60:.1f} min')"))"

if [ $exit_code -eq 0 ]; then
    echo "✅ Author-level DPSGD Training completed successfully!"
    echo "📁 Model saved to $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
else
    echo "❌ Author-level DPSGD Training failed with exit code: $exit_code"
    echo "🔧 TROUBLESHOOTING:"
    echo "   Check log files for specific errors:"
    echo "   - Full Training Log: logs/simple_lora_finetune_author_level_DPSGD_output_${TIMESTAMP}.log"
fi

echo ""
echo "🏁 Script Finished: $(date)"

exit $exit_code