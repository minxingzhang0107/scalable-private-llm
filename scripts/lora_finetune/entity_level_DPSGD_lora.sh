#!/bin/bash

set -e

# =============================================================================
# User-Level Differential Privacy LoRA Training
# FINAL CORRECTED IMPLEMENTATION - All Bugs Fixed
# =============================================================================

echo "🚀 User-Level DP LoRA Training Started: $(date)"
echo "✅ FINAL CORRECTED VERSION with all fixes applied"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HOME/.cache/huggingface"

# Clear GPU memory
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
OUTPUT_DIR="./model_checkpoints/entity_level_DPSGD_lora_mistral_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# DP CONFIGURATION
# =============================================================================

EPSILON=10.0                    # Privacy budget epsilon
DELTA=1e-5                      # Privacy budget delta
MAX_GRAD_NORM=3.0               # Gradient clipping norm C
USER_SAMPLING_RATE=0.01         # Sample 1% of users per step

# GRADIENT NORM AUTO-TUNING
# Set to "true" to monitor gradient norms and recommend optimal C
AUTO_TUNE_CLIPPING=true
CLIPPING_PERCENTILE=50          # Use median (50th percentile) for C recommendation

# IMPORTANT NOTES ON CLIPPING NORM C:
# - C=1.0 is just a default starting point, not necessarily optimal
# - Choosing C involves a trade-off:
#   * Small C: Less noise needed, but more information loss from clipping
#   * Large C: Preserves gradient information, but requires more noise
# - Best practice: Set AUTO_TUNE_CLIPPING=true to observe your gradient norms
#   and tune C to a percentile (typically median or 75th percentile)
# - If most gradients are much larger than C, you're losing information
# - If most gradients are much smaller than C, you could use smaller C and less noise

# IMPORTANT NOTES ON DELTA:
# - Delta should be < 1/n where n is the number of users
# - For 200 users, delta should be < 0.005 (we use 1e-5 = 0.00001 ✓)
# - The code will validate this and raise an error if delta is too large

# IMPORTANT NOTES ON TRAINING STEPS:
# - With q=0.01, each epoch = ceil(1/0.01) = 100 steps (CORRECTED)
# - For 3 epochs, total training steps = 300 steps (CORRECTED)
# - Privacy accounting is based on these 300 steps (CORRECTED)
# - This is the KEY FIX from Gemini's feedback!

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

EPOCHS=3
LEARNING_RATE=1e-4              # Learning rate
MAX_LENGTH=256                  # Max sequence length

# =============================================================================
# LORA CONFIGURATION
# =============================================================================

LORA_R=8
LORA_ALPHA=8
LORA_DROPOUT=0.1

# =============================================================================
# PRE-RUN CHECKS
# =============================================================================

PYTHON_SCRIPT="src/lora_finetune/entity_level_DPSGD_lora.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Missing Python script: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Dataset file not found at $DATA_FILE"
    exit 1
fi

mkdir -p logs

echo "🧮 System check..."
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

echo "✅ Pre-run checks completed"

# =============================================================================
# CALCULATE AND DISPLAY TRAINING PARAMETERS
# =============================================================================

STEPS_PER_EPOCH=$(python -c "import math; print(math.ceil(1.0 / $USER_SAMPLING_RATE))")
TOTAL_STEPS=$(python -c "print($EPOCHS * $STEPS_PER_EPOCH)")

echo ""
echo "============================================================"
echo "TRAINING CONFIGURATION (FINAL CORRECTED VERSION)"
echo "============================================================"
echo ""
echo "   PRIVACY PARAMETERS:"
echo "   - Privacy budget: (epsilon=$EPSILON, delta=$DELTA)"
echo "   - Clipping norm C: $MAX_GRAD_NORM"
if [ "$AUTO_TUNE_CLIPPING" = "true" ]; then
echo "   - Auto-tuning: ENABLED (will recommend optimal C)"
echo "   - Percentile: ${CLIPPING_PERCENTILE}th"
else
echo "   - Auto-tuning: DISABLED (using fixed C)"
fi
echo "   - User sampling rate q: $USER_SAMPLING_RATE"
echo "   - This provides USER-LEVEL DP (not instance-level)"
echo ""
echo "   📅 TRAINING SCHEDULE (CORRECTED):"
echo "   - Epochs: $EPOCHS"
echo "   - Steps per epoch: $STEPS_PER_EPOCH (= ceil(1/q)) ✓ FIXED"
echo "   - Total training steps: $TOTAL_STEPS ✓ FIXED"
echo "   - Learning rate: $LEARNING_RATE"
echo "   - LR decay: 0.9 per epoch ✓ FIXED (not per step)"
echo ""
echo "   🔧 MODEL CONFIGURATION:"
echo "   - Model: $MODEL_NAME"
echo "   - Data: $DATA_FILE"
echo "   - Max sequence length: $MAX_LENGTH"
echo ""
echo "   🎯 LORA PARAMETERS:"
echo "   - Rank r: $LORA_R"
echo "   - Alpha: $LORA_ALPHA"
echo "   - Dropout: $LORA_DROPOUT"
echo ""
echo "   ✅ ALL CRITICAL FIXES APPLIED:"
echo "   - ✅ FIX #1: Delta validation (δ < 1/n)"
echo "   - ✅ FIX #2: Correct privacy accounting formula"
echo "   - ✅ FIX #3: LR decay per epoch, not per step"
echo "   - ✅ FIX #4: Correct epoch definition (ceil(1/q))"
echo "   - ✅ FIX #5: Privacy spent tracking and output"
echo "   - ✅ Processes ALL examples from each sampled user"
echo "   - ✅ No within-user subsampling"
echo "   - ✅ Correct gradient flow: sum → noise → average"
echo "   - ✅ Proper noise scaling: sensitivity = C"
echo ""
echo "   🎓 THIS IS USER-LEVEL DP:"
echo "   - Privacy guarantee: adding/removing ANY single user"
echo "     (with all their data) changes the model by ≤ε w.p. 1-δ"
echo "   - NOT instance-level DP (which is about single examples)"
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
CMD_ARGS="$CMD_ARGS --lora-r $LORA_R"
CMD_ARGS="$CMD_ARGS --lora-alpha $LORA_ALPHA"
CMD_ARGS="$CMD_ARGS --lora-dropout $LORA_DROPOUT"
CMD_ARGS="$CMD_ARGS --clipping-percentile $CLIPPING_PERCENTILE"

# Add auto-tuning flag if enabled
if [ "$AUTO_TUNE_CLIPPING" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --auto-tune-clipping"
fi

# =============================================================================
# RUN TRAINING
# =============================================================================

echo ""
echo "🚀 Starting User-Level DP Training..."
echo "============================================================"

start_time=$(date +%s)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python -u "$PYTHON_SCRIPT" $CMD_ARGS 2>&1 | tee -a "logs/entity_level_DPSGD_lora_training_${TIMESTAMP}.log"
exit_code=${PIPESTATUS[0]}

end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$(python -c "print(f'{$duration/60:.1f}')")

echo "============================================================"
echo "📅 Completed: $(date)"
echo "⏱️  Duration: ${duration}s ($minutes min)"

if [ $exit_code -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    echo ""
    echo "📊 Output files:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    echo "🔒 PRIVACY GUARANTEE:"
    echo "   Privacy spent: (ε=$EPSILON, δ=$DELTA)"
    echo "   Based on $TOTAL_STEPS training steps"
    echo "   User-level DP: Adding/removing any user's data"
    echo "   changes model output by at most ε with probability 1-δ"
else
    echo "❌ Training failed with exit code: $exit_code"
    echo ""
    echo "🔧 Check logs:"
    echo "   - Training: logs/entity_level_DPSGD_lora_training_${TIMESTAMP}.log"
fi

echo ""
echo "🏁 Finished: $(date)"
echo "============================================================"

exit $exit_code