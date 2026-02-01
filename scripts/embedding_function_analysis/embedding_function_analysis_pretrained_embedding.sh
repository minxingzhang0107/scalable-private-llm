#!/bin/bash

# =============================================================================
# Combined LM+1NN Dynamic Lambda KNN-LM Generation Pipeline with DATASTORE REMOVAL
# LM: Pre-trained Mistral 7B | Embeddings: Pre-trained Mistral 7B
# Task: 2b-Combined1NN (DYNAMIC λ) with PRE-TRAINED EMBEDDINGS + DATASTORE REMOVAL
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

TEST_FILE="dataset/private/tofu/tofu_test.json"

# KNN-LM parameters  
K=1                         # Number of neighbors for KNN (used for building datastore)
BATCH_SIZE=256             # Batch size for A6000

# 🔥 DYNAMIC LAMBDA PARAMETERS
UPPER_LAMBDA=1.0           # High KNN weight when distance < threshold (close neighbors)
LOWER_LAMBDA=0.0           # Low KNN weight when distance >= threshold (far neighbors)  
DISTANCE_THRESHOLD=0.4     # Distance threshold for lambda assignment

# Output file for generated answers - Updated to indicate removal functionality
# OUTPUT_FILE="privacy_analysis/1nn_ds_only_pretrained_embedding_with_removal_generated_answers.json"
OUTPUT_FILE="results/embedding_function_analysis/combined_pretrained_embedding_with_removal_generated_answers_distance_0.4.json"


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 Combined LM+1NN Pre-trained Embedding Dynamic Lambda KNN-LM Configuration with DATASTORE REMOVAL:"
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
echo "     2b: Combined LM + 1NN (DYNAMIC λ) with PRE-TRAINED EMBEDDINGS + DATASTORE REMOVAL"
echo "   🤖 LM: Pre-trained Mistral 7B"
echo "   🎯 Embeddings: Pre-trained Mistral 7B"
echo "   🗑️ Datastore Removal: Enabled (per-query removal by original_index)"

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
# RUN COMBINED LM+1NN DYNAMIC LAMBDA KNN-LM GENERATION WITH PRE-TRAINED EMBEDDINGS + DATASTORE REMOVAL
# =============================================================================

echo "🚀 Starting Combined LM+1NN Dynamic Lambda KNN-LM Generation Pipeline with PRE-TRAINED EMBEDDINGS + DATASTORE REMOVAL..."
echo "⏰ NO TIME LIMIT - Will run until completion!"

python src/embedding_function_analysis/embedding_function_analysis_pretrained_embedding.py \
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
    echo "✅ Combined LM+1NN Dynamic Lambda KNN-LM generation with PRE-TRAINED EMBEDDINGS + DATASTORE REMOVAL completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Task 2b: Combined LM+1NN (DYNAMIC λ) with PRE-TRAINED EMBEDDINGS + DATASTORE REMOVAL completed"
    echo "   🤖 LM Model: Pre-trained Mistral 7B"
    echo "   🎯 Embedding Model: Pre-trained Mistral 7B"
    echo "   🗑️ Datastore Removal: Enabled - removes QA pairs by original_index during testing"
    echo "   🔥 Dynamic λ rule:"
    echo "     - Distance < $DISTANCE_THRESHOLD → λ = $UPPER_LAMBDA (high KNN weight)"
    echo "     - Distance >= $DISTANCE_THRESHOLD → λ = $LOWER_LAMBDA (low KNN weight)"
    echo "   💾 Generated answers saved to: $OUTPUT_FILE"
    
    # Show some stats if output file exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "📊 Output file stats:"
        echo "   Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo "   Questions processed: $(python -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))" 2>/dev/null || echo "Unknown")"
    fi
else
    echo "❌ Combined LM+1NN Dynamic Lambda KNN-LM generation with PRE-TRAINED EMBEDDINGS + DATASTORE REMOVAL failed!"
    exit 1
fi

echo "🎯 Pre-trained embedding pipeline with datastore removal complete!"