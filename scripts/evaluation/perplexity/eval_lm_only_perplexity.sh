#!/bin/bash

# =============================================================================
# Perplexity Evaluation Pipeline
# Task: Compare pretrained vs fine-tuned model perplexity
# =============================================================================

set -e

# Create logs directory
mkdir -p logs


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
# PARAMETERS
# =============================================================================

# Fine-tuned model path - CHANGE THIS TO YOUR ACTUAL FINE-TUNED MODEL PATH
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152332"
ADAPTER_PATH="./model_checkpoints/user_dp_lora_mistral_20251003_203637"

# Test data file (same as original scripts)
TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"
# TEST_FILE="dataset/public_test_tiny_qa.json"

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