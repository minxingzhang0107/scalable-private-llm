#!/bin/bash

set -e

# Create logs directory
mkdir -p logs

# =============================================================================
# CONFIGURATION: Model and Dataset Selection
# =============================================================================
# Uncomment ONE model configuration and ONE test dataset based on your experiment

# --- MODEL SELECTION ---
# Option 1: Model trained on both PUBLIC data and PRIVATE data (without any privacy protection)
ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152332"

# Option 2: Model trained on PUBLIC and PRIVATE data (with DPSGD privacy protection)
# ADAPTER_PATH="./model_checkpoints/user_dp_lora_mistral_20251003_203637"

# --- TEST DATASET SELECTION ---
# Option A: Test on PRIVATE data (TOFU dataset)
TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"

# Option B: Test on PUBLIC data
# TEST_FILE="dataset/public/public_test_tiny_qa.json"

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================
#
# Experiment 1: Public model on Private data perplexity & Public + Private model on Private data perplexity
#   - Use: ADAPTER_PATH (Option 1) + TEST_FILE (Option A)
#   - Measures: Two model perplexities on private data
#
# Experiment 2: Public model on Public data perplexity & Public + Private model on Public data perplexity
#   - Use: ADAPTER_PATH (Option 1) + TEST_FILE (Option B)
#   - Measures: Two model perplexities on public data
#
# Experiment 3: Public model on Private data perplexity & Public + Private (with DPSGD) model on Private data perplexity
#   - Use: ADAPTER_PATH (Option 2) + TEST_FILE (Option A)
#   - Measures: Two model perplexities on private data
#
# Experiment 4: Public model on Public data perplexity & Public + Private (with DPSGD) model on Public data perplexity
#   - Use: ADAPTER_PATH (Option 2) + TEST_FILE (Option B)
#   - Measures: Two model perplexities on public data
#
# =============================================================================

echo "🖥️ GPU Configuration:"
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
else:
    print('❌ CUDA not available')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 Perplexity Evaluation Configuration:"
echo "   Test file: $TEST_FILE"
echo "   Fine-tuned adapter: $ADAPTER_PATH"
echo "   🎯 TASK: Compare pretrained vs fine-tuned model perplexity"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

# Check if fine-tuned model exists (optional)
if [ -d "$ADAPTER_PATH" ]; then
    echo "✅ Fine-tuned adapter found: $ADAPTER_PATH"
    
    # Check for key adapter files
    if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
        echo "❌ Missing adapter_config.json in: $ADAPTER_PATH"
        exit 1
    fi
    
    if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ] && [ ! -f "$ADAPTER_PATH/adapter_model.bin" ]; then
        echo "❌ Missing adapter model files in: $ADAPTER_PATH"
        exit 1
    fi
    
    echo "✅ Fine-tuned adapter validated"
    USE_FINETUNED="--adapter-path $ADAPTER_PATH"
else
    echo "⚠️ Fine-tuned adapter not found: $ADAPTER_PATH"
    echo "   Will only evaluate pretrained model"
    USE_FINETUNED=""
fi

echo "✅ All parameters validated"

# =============================================================================
# RUN PERPLEXITY EVALUATION
# =============================================================================

echo "🚀 Starting Perplexity Evaluation Pipeline..."
echo "⏰ NO TIME LIMIT - Will run until completion!"

# Run the perplexity evaluation
python src/evaluation/perplexity/eval_lm_only_perplexity.py \
    --test-file "$TEST_FILE" \
    $USE_FINETUNED

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Perplexity evaluation completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Pretrained model perplexity: Calculated"
    if [ ! -z "$USE_FINETUNED" ]; then
        echo "   ✅ Fine-tuned model perplexity: Calculated"
        echo "   📊 Comparison: See output above"
    else
        echo "   ⚠️ Fine-tuned model: Not evaluated"
    fi
else
    echo "❌ Perplexity evaluation failed!"
    exit 1
fi

echo "🎯 Perplexity evaluation pipeline complete!"