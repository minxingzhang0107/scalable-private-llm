#!/bin/bash

# =============================================================================
# LM-Only Generation Pipeline with TIMING
# Task: 1a-LM ONLY - Inference timing (NO ANSWER FILE SAVING)
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
TEST_FILE="dataset/private/syn_traj/scalability_dataset/test_1k.json"
# TEST_FILE="dataset/public/public_test_tiny_qa.json"

# NO OUTPUT FILE - Just measuring inference times

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 LM-Only Generation with Timing Configuration:"
echo "   Test file: $TEST_FILE"
echo "   🎯 TASK:"
echo "     1a: LM-only generation with inference timing"
echo "     📊 Will report average and std dev of inference times"
echo "     💾 No answer file will be saved"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

echo "✅ All parameters validated"

# =============================================================================
# RUN LM-ONLY GENERATION WITH TIMING
# =============================================================================

echo "🚀 Starting LM-Only Generation with Timing Pipeline..."
echo "⏰ NO TIME LIMIT - Will run until completion!"
echo "⏱️ Measuring inference time for each query..."

python src/plot/lm_only_generation_time.py \
    --test-file "$TEST_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ LM-Only timing evaluation completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Task 1a: LM-only generation with timing completed"
    echo "   📊 Timing statistics reported above"
    echo "   💾 No answer file saved (timing only)"
else
    echo "❌ LM-Only timing evaluation failed!"
    exit 1
fi

echo "🎯 Pipeline complete!"