#!/bin/bash

# =============================================================================
# LM-Only Generation Pipeline
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
# PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Data files (NO TRAIN FILE - NO DATASTORE BUILDING)
TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"
# TEST_FILE="dataset/public/public_test_tiny_qa.json"

# Output file for generated answers
OUTPUT_FILE="results/private/tofu/lm_only/pretrained_lm_only_generated_answers.json"
# OUTPUT_FILE="results/public/lm_only/pretrained_lm_only_generated_answers.json"

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 LM-Only Generation Configuration:"
echo "   Test file: $TEST_FILE"
echo "   Output file: $OUTPUT_FILE"
echo "   🎯 TASK:"
echo "     1a: LM-only generation (NO DATASTORE)"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "✅ All parameters validated"

# =============================================================================
# RUN LM-ONLY GENERATION
# =============================================================================

echo "🚀 Starting LM-Only Generation Pipeline..."
echo "⏰ NO TIME LIMIT - Will run until completion!"

python src/generation/generation_lm_only_qa.py \
    --test-file "$TEST_FILE" \
    --output-file "$OUTPUT_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ LM-Only generation completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Task 1a: LM-only generation completed"
    echo "   💾 Generated answers saved to: $OUTPUT_FILE"
else
    echo "❌ LM-Only generation failed!"
    exit 1
fi

echo "🎯 Pipeline complete!"