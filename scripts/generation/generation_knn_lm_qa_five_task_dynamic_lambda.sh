#!/bin/bash

# =============================================================================
# Five-Task Dynamic Lambda KNN-LM Generation Pipeline
# Tasks: 1a-LM | 1b-1NN | 1c-KNN | 2a-Combined | 2b-Combined1NN (DYNAMIC λ)
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

# KNN-LM parameters  
K=9                         # Number of neighbors for KNN (used in tasks 1c and 2a)
BATCH_SIZE=256             # Batch size for A6000

# 🔥 DYNAMIC LAMBDA PARAMETERS
UPPER_LAMBDA=1.0           # High KNN weight when distance < threshold (close neighbors)
LOWER_LAMBDA=0.0           # Low KNN weight when distance >= threshold (far neighbors)  
DISTANCE_THRESHOLD=0.3     # Distance threshold for lambda assignment

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 Five-Task Dynamic Lambda KNN-LM Configuration:"
echo "   Train file: $TRAIN_FILE"
echo "   Test file: $TEST_FILE"
echo "   K neighbors: $K"
echo "   Batch size: $BATCH_SIZE"
echo "   🔥 DYNAMIC LAMBDA SETTINGS:"
echo "   Upper lambda (near): $UPPER_LAMBDA"
echo "   Lower lambda (far): $LOWER_LAMBDA"
echo "   Distance threshold: $DISTANCE_THRESHOLD"
echo "   🎯 FIVE TASKS:"
echo "     1a: LM-only generation"
echo "     1b: 1NN-only generation"
echo "     1c: KNN-only generation (k=$K neighbors)"
echo "     2a: Combined LM + KNN (DYNAMIC λ)"
echo "     2b: Combined LM + 1NN (DYNAMIC λ)"

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
# RUN FIVE-TASK DYNAMIC LAMBDA KNN-LM GENERATION
# =============================================================================

echo "🚀 Starting Five-Task Dynamic Lambda KNN-LM Generation Pipeline..."
echo "⏰ NO TIME LIMIT - Will run until completion!"

python src/generation/generation_knn_lm_qa_five_task_dynamic_lambda.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --k $K \
    --upper-lambda $UPPER_LAMBDA \
    --lower-lambda $LOWER_LAMBDA \
    --distance-threshold $DISTANCE_THRESHOLD \
    --batch-size $BATCH_SIZE

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Five-Task Dynamic Lambda KNN-LM generation completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Task 1a: LM-only generation completed"
    echo "   ✅ Task 1b: 1NN-only generation completed"
    echo "   ✅ Task 1c: KNN-only generation (k=$K) completed"
    echo "   ✅ Task 2a: Combined LM+KNN (DYNAMIC λ) completed"
    echo "   ✅ Task 2b: Combined LM+1NN (DYNAMIC λ) completed"
    echo "   🔥 Dynamic λ rule:"
    echo "     - Distance < $DISTANCE_THRESHOLD → λ = $UPPER_LAMBDA (high KNN weight)"
    echo "     - Distance >= $DISTANCE_THRESHOLD → λ = $LOWER_LAMBDA (low KNN weight)"
else
    echo "❌ Five-Task Dynamic Lambda KNN-LM generation failed!"
    exit 1
fi

echo "🎯 Pipeline complete!"