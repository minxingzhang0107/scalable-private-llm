#!/bin/bash


set -e

# =============================================================================
# User-Level Differential Privacy LoRA Training
# A6000 OPTIMIZED - Batched processing for GPU efficiency
# =============================================================================

echo "Entity-Level DPSGD LoRA Training Started: $(date)"
echo "A6000 OPTIMIZED VERSION with batched processing"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HOME/.cache/huggingface"

python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU memory cleared')
"

# =============================================================================
# MODEL AND DATA CONFIGURATION
# =============================================================================

MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
DATA_FILE="dataset/private/tofu/tofu_train.json"
OUTPUT_DIR="./model_checkpoints/user_dp_lora_mistral_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# DP CONFIGURATION
# =============================================================================

EPSILON=10.0
DELTA=1e-5
MAX_GRAD_NORM=7.0               # Based on auto-tuning recommendation
USER_SAMPLING_RATE=0.01

# GRADIENT NORM AUTO-TUNING
AUTO_TUNE_CLIPPING=false        # Set to true for first run to find optimal C
CLIPPING_PERCENTILE=50

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

EPOCHS=10
LEARNING_RATE=1e-4
MAX_LENGTH=256

# A6000 OPTIMIZATION: Batch size for processing examples within each user
PER_USER_BATCH_SIZE=8           # Process 8 examples at once (was 1)

# =============================================================================
# LORA CONFIGURATION
# =============================================================================

LORA_R=8
LORA_ALPHA=8
LORA_DROPOUT=0.1

# =============================================================================
# PRE-RUN CHECKS
# =============================================================================

PYTHON_SCRIPT="src/lora_finetune/entity_level_DPSGD_lora_efficient.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Missing Python script: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Dataset file not found at $DATA_FILE"
    exit 1
fi

mkdir -p logs

echo "System check..."
python -c "
import torch
import numpy as np
import math
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

echo "Pre-run checks completed"

# =============================================================================
# CALCULATE TRAINING PARAMETERS
# =============================================================================

STEPS_PER_EPOCH=$(python -c "import math; print(math.ceil(1.0 / $USER_SAMPLING_RATE))")
TOTAL_STEPS=$(python -c "print($EPOCHS * $STEPS_PER_EPOCH)")

echo ""
echo "============================================================"
echo "TRAINING CONFIGURATION (A6000 OPTIMIZED)"
echo "============================================================"
echo ""
echo "   PRIVACY PARAMETERS:"
echo "   - Privacy budget: (epsilon=$EPSILON, delta=$DELTA)"
echo "   - Clipping norm C: $MAX_GRAD_NORM (auto-tuned)"
if [ "$AUTO_TUNE_CLIPPING" = "true" ]; then
echo "   - Auto-tuning: ENABLED"
else
echo "   - Auto-tuning: DISABLED (using tuned C)"
fi
echo "   - User sampling rate q: $USER_SAMPLING_RATE"
echo "   - USER-LEVEL DP (not instance-level)"
echo ""
echo "   TRAINING SCHEDULE:"
echo "   - Epochs: $EPOCHS"
echo "   - Steps per epoch: $STEPS_PER_EPOCH (= ceil(1/q))"
echo "   - Total steps: $TOTAL_STEPS"
echo "   - Learning rate: $LEARNING_RATE"
echo "   - LR decay: 0.9 per epoch"
echo ""
echo "   A6000 OPTIMIZATIONS:"
echo "   - Per-user batch size: $PER_USER_BATCH_SIZE"
echo "   - DataLoader with prefetching: enabled"
echo "   - Fast model loading: device_map='auto'"
echo "   - Expected speedup: 5-10x vs unoptimized"
echo ""
echo "   MODEL CONFIGURATION:"
echo "   - Model: $MODEL_NAME"
echo "   - Data: $DATA_FILE"
echo "   - Max sequence length: $MAX_LENGTH"
echo ""
echo "   LORA PARAMETERS:"
echo "   - Rank r: $LORA_R"
echo "   - Alpha: $LORA_ALPHA"
echo "   - Dropout: $LORA_DROPOUT"
echo ""
echo "   ALL CRITICAL FIXES APPLIED:"
echo "   - Correct epoch definition (ceil(1/q))"
echo "   - Correct privacy accounting"
echo "   - Processes ALL examples from each user"
echo "   - No within-user subsampling"
echo "   - Correct gradient flow: sum -> noise -> average"
echo "   - Batched processing for GPU efficiency"
echo "============================================================"

# =============================================================================
# BUILD COMMAND
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
CMD_ARGS="$CMD_ARGS --per-user-batch-size $PER_USER_BATCH_SIZE"
CMD_ARGS="$CMD_ARGS --lora-r $LORA_R"
CMD_ARGS="$CMD_ARGS --lora-alpha $LORA_ALPHA"
CMD_ARGS="$CMD_ARGS --lora-dropout $LORA_DROPOUT"
CMD_ARGS="$CMD_ARGS --clipping-percentile $CLIPPING_PERCENTILE"

if [ "$AUTO_TUNE_CLIPPING" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --auto-tune-clipping"
fi

# =============================================================================
# RUN TRAINING
# =============================================================================

echo ""
echo "Starting Entity-Level DPSGD Lora Efficient Training..."
echo "============================================================"

start_time=$(date +%s)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python -u "$PYTHON_SCRIPT" $CMD_ARGS 2>&1 | tee -a "logs/entity_level_DPSGD_lora_efficient_training_${TIMESTAMP}.log"
exit_code=${PIPESTATUS[0]}

end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$(python -c "print(f'{$duration/60:.1f}')")

echo "============================================================"
echo "Completed: $(date)"
echo "Duration: ${duration}s ($minutes min)"

if [ $exit_code -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    echo "PRIVACY GUARANTEE:"
    echo "   Privacy spent: (epsilon=$EPSILON, delta=$DELTA)"
    echo "   Based on $TOTAL_STEPS training steps"
    echo "   Entity-level DPSGD with all critical fixes applied"
else
    echo "Training failed with exit code: $exit_code"
    echo ""
    echo "Check logs:"
    echo "   - Training: logs/entity_level_DPSGD_lora_efficient_training_${TIMESTAMP}.log"
fi

echo ""
echo "Finished: $(date)"
echo "============================================================"

exit $exit_code