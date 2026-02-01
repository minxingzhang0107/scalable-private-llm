#!/bin/bash

# =============================================================================
# KNN-LM Perplexity Evaluation Pipeline (FIXED VERSION)
# Task: Evaluate perplexity with Combined LM+1NN (DYNAMIC λ) - Fixed Tokenization
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

# Data files
TRAIN_FILE="dataset/private/tofu/tofu_train.json"

TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"
# TEST_FILE="dataset/public/public_test_tiny_qa.json"

# KNN-LM parameters  
K=1                         # Number of neighbors for KNN (used for building datastore)
BATCH_SIZE=256             # Batch size for A6000

# 🔥 DYNAMIC LAMBDA PARAMETERS
UPPER_LAMBDA=1.0           # High KNN weight when distance < threshold (close neighbors)
LOWER_LAMBDA=0.0           # Low KNN weight when distance >= threshold (far neighbors)  
DISTANCE_THRESHOLD=0.4     # Distance threshold for lambda assignment (updated to match fixed script)

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 KNN-LM Perplexity Evaluation Configuration (FIXED VERSION):"
echo "   Train file: $TRAIN_FILE"
echo "   Test file: $TEST_FILE"
echo "   K neighbors: $K"
echo "   Batch size: $BATCH_SIZE"
echo "   🔥 DYNAMIC LAMBDA SETTINGS:"
echo "   Upper lambda (near): $UPPER_LAMBDA"
echo "   Lower lambda (far): $LOWER_LAMBDA"
echo "   Distance threshold: $DISTANCE_THRESHOLD"
echo "   🎯 TASK: KNN-LM Perplexity Evaluation with Combined LM + 1NN (DYNAMIC λ)"
echo "   🔧 FIXED: Uses same tokenization as LM-only baseline"

# Check if files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Train file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

# Validate K parameter
if [ $K -lt 1 ] || [ $K -gt 100 ]; then
    echo "❌ Invalid K value: $K (should be 1-100)"
    exit 1
fi

# Validate Lambda parameters
if (( $(echo "$UPPER_LAMBDA < 0.0" | bc -l) )) || (( $(echo "$UPPER_LAMBDA > 1.0" | bc -l) )); then
    echo "❌ Invalid Upper Lambda: $UPPER_LAMBDA (should be 0.0-1.0)"
    exit 1
fi

if (( $(echo "$LOWER_LAMBDA < 0.0" | bc -l) )) || (( $(echo "$LOWER_LAMBDA > 1.0" | bc -l) )); then
    echo "❌ Invalid Lower Lambda: $LOWER_LAMBDA (should be 0.0-1.0)"
    exit 1
fi

# Validate Distance Threshold
if (( $(echo "$DISTANCE_THRESHOLD < 0.0" | bc -l) )) || (( $(echo "$DISTANCE_THRESHOLD > 2.0" | bc -l) )); then
    echo "❌ Invalid Distance Threshold: $DISTANCE_THRESHOLD (should be 0.0-2.0)"
    exit 1
fi

echo "✅ All parameters validated"

# =============================================================================
# RUN KNN-LM PERPLEXITY EVALUATION (FIXED VERSION)
# =============================================================================

echo "🚀 Starting KNN-LM Perplexity Evaluation Pipeline (FIXED)..."
echo "⏰ NO TIME LIMIT - Will run until completion!"
echo "🔧 This fixed version uses same tokenization as LM-only baseline"

# CHANGE: Updated script name to fixed version
python src/evaluation/perplexity/eval_knn_lm_perplexity_dynamic_lambda.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --k $K \
    --upper-lambda $UPPER_LAMBDA \
    --lower-lambda $LOWER_LAMBDA \
    --distance-threshold $DISTANCE_THRESHOLD \
    --batch-size $BATCH_SIZE

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ KNN-LM Perplexity Evaluation (FIXED) completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ KNN-LM Perplexity with Combined LM+1NN (DYNAMIC λ) calculated"
    echo "   🔥 Dynamic λ rule:"
    echo "     - Distance < $DISTANCE_THRESHOLD → λ = $UPPER_LAMBDA (high KNN weight)"
    echo "     - Distance >= $DISTANCE_THRESHOLD → λ = $LOWER_LAMBDA (low KNN weight)"
    echo "   🔧 FIXED: Should now match LM-only baseline when λ=0.0"
    echo "   📊 See perplexity results above"
else
    echo "❌ KNN-LM Perplexity Evaluation (FIXED) failed!"
    exit 1
fi

echo "🎯 Fixed perplexity evaluation pipeline complete!"