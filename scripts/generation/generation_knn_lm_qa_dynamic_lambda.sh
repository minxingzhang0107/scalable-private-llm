#!/bin/bash

# =============================================================================
# CONFIGURATION: Dataset, Distance Threshold, and Output Selection
# =============================================================================
# --- KNN-LM PARAMETERS ---
DISTANCE_THRESHOLD=0.4     # Distance threshold for dynamic lambda (options: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8). Just directly modify this value

# --- TRAIN DATASET ---
TRAIN_FILE="dataset/private/tofu/tofu_train.json"

# --- TEST DATASET AND OUTPUT SELECTION (uncomment ONE option) ---
# Option A: Test on PRIVATE data
# TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"
# OUTPUT_FILE="results/private/tofu/1nn_lm_pretrained_embedding/combined_1nn_lm_generated_answers_pretrained_embedding_threshold_${DISTANCE_THRESHOLD}.json"

# Option B: Test on PUBLIC data
TEST_FILE="dataset/public/public_test_tiny_qa.json"
OUTPUT_FILE="results/public/1nn_lm_pretrained_embedding/combined_1nn_lm_generated_answers_threshold_${DISTANCE_THRESHOLD}.json"


# NOTE: The output filename automatically includes the distance threshold value.
#       When you change DISTANCE_THRESHOLD, the filename updates accordingly.
#       Example: DISTANCE_THRESHOLD=0.4 → "...threshold_0.4.json"

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
# KNN-LM parameters  
K=1                         # Number of neighbors for KNN (used for building datastore)
BATCH_SIZE=256             # Batch size for A6000

# 🔥 DYNAMIC LAMBDA PARAMETERS
UPPER_LAMBDA=1           # High KNN weight when distance < threshold (close neighbors)
LOWER_LAMBDA=0           # Low KNN weight when distance >= threshold (far neighbors)  
# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 Combined LM+1NN Dynamic Lambda KNN-LM Configuration:"
echo "   Train file: $TRAIN_FILE"
echo "   Test file: $TEST_FILE"
echo "   Output file: $OUTPUT_FILE"
echo "   K neighbors: $K"
echo "   Batch size: $BATCH_SIZE"
echo "   🔥 DYNAMIC LAMBDA SETTINGS:"
echo "   Upper lambda (near): $UPPER_LAMBDA"
echo "   Lower lambda (far): $LOWER_LAMBDA"
echo "   Distance threshold: $DISTANCE_THRESHOLD"
echo "   🎯 TASK:"
echo "     2b: Combined LM + 1NN (DYNAMIC λ) ONLY"

# Check if files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Train file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

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
# RUN COMBINED LM+1NN DYNAMIC LAMBDA KNN-LM GENERATION
# =============================================================================

echo "🚀 Starting Combined LM+1NN Dynamic Lambda KNN-LM Generation Pipeline..."
echo "⏰ NO TIME LIMIT - Will run until completion!"

python src/generation/generation_knn_lm_qa_dynamic_lambda.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --output-file "$OUTPUT_FILE" \
    --k $K \
    --upper-lambda $UPPER_LAMBDA \
    --lower-lambda $LOWER_LAMBDA \
    --distance-threshold $DISTANCE_THRESHOLD \
    --batch-size $BATCH_SIZE

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Combined LM+1NN Dynamic Lambda KNN-LM generation completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Task 2b: Combined LM+1NN (DYNAMIC λ) completed"
    echo "   🔥 Dynamic λ rule:"
    echo "     - Distance < $DISTANCE_THRESHOLD → λ = $UPPER_LAMBDA (high KNN weight)"
    echo "     - Distance >= $DISTANCE_THRESHOLD → λ = $LOWER_LAMBDA (low KNN weight)"
    echo "   💾 Generated answers saved to: $OUTPUT_FILE"
else
    echo "❌ Combined LM+1NN Dynamic Lambda KNN-LM generation failed!"
    exit 1
fi

echo "🎯 Pipeline complete!"