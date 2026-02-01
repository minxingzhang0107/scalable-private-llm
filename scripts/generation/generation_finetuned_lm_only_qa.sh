#!/bin/bash

# =============================================================================
# LM-Only Generation Pipeline WITH FINE-TUNED MODEL
# Task: 1a-LM ONLY (NO DATASTORE BUILDING)
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
# PARAMETERS - FINE-TUNED MODEL
# =============================================================================

# Fine-tuned model path - CHANGE THIS TO YOUR ACTUAL FINE-TUNED MODEL PATH
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152332"
# ADAPTER_PATH="./model_checkpoints/simple_lora_dp_mistral_20250804_181933" 
ADAPTER_PATH="./model_checkpoints/user_dp_lora_mistral_20251003_203637" 

# Data files (NO TRAIN FILE - NO DATASTORE BUILDING)
# TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"
TEST_FILE="dataset/public/public_test_tiny_qa.json"

# Output file for generated answers
# OUTPUT_FILE="results/private/tofu/lm_only/finetuned_lm_only_generated_answers.json"
# OUTPUT_FILE="results/private/tofu/lm_only/DPSGD_tuned_lm_only_generated_answers.json"

# OUTPUT_FILE="results/public/lm_only/finetuned_lm_only_generated_answers.json"
OUTPUT_FILE="results/public/lm_only/DPSGD_tuned_lm_only_generated_answers.json"

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 LM-Only Generation Configuration (FINE-TUNED):"
echo "   Fine-tuned adapter: $ADAPTER_PATH"
echo "   Test file: $TEST_FILE"
echo "   Output file: $OUTPUT_FILE"
echo "   🎯 TASK:"
echo "     1a: LM-only generation with FINE-TUNED MODEL (NO DATASTORE)"

# Check if fine-tuned model exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ Fine-tuned adapter not found: $ADAPTER_PATH"
    echo "   Make sure your LoRA fine-tuning completed successfully"
    exit 1
fi

# Check for key adapter files
if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
    echo "❌ Missing adapter_config.json in: $ADAPTER_PATH"
    exit 1
fi

if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ] && [ ! -f "$ADAPTER_PATH/adapter_model.bin" ]; then
    echo "❌ Missing adapter model files in: $ADAPTER_PATH"
    exit 1
fi

echo "✅ Fine-tuned adapter found and validated"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "✅ All parameters validated"

# =============================================================================
# RUN LM-ONLY GENERATION WITH FINE-TUNED MODEL
# =============================================================================

echo "🚀 Starting LM-Only Generation Pipeline with FINE-TUNED MODEL..."
echo "⏰ NO TIME LIMIT - Will run until completion!"

python src/generation/generation_finetuned_lm_only_qa.py \
    --test-file "$TEST_FILE" \
    --output-file "$OUTPUT_FILE" \
    --adapter-path "$ADAPTER_PATH"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ LM-Only generation with fine-tuned model completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Task 1a: LM-only generation with FINE-TUNED MODEL completed"
    echo "   🎯 Fine-tuned adapter: $ADAPTER_PATH"
    echo "   💾 Generated answers saved to: $OUTPUT_FILE"
    
    # Show some stats if output file exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "📊 Output file stats:"
        echo "   Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo "   Questions processed: $(python -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))" 2>/dev/null || echo "Unknown")"
    fi
else
    echo "❌ LM-Only generation with fine-tuned model failed!"
    exit 1
fi

echo "🎯 Fine-tuned pipeline complete!"