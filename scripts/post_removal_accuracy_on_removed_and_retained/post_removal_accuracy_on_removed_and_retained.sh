#!/bin/bash

# =============================================================================
# KNN-LM Generation Pipeline with PERMANENT AUTHOR REMOVAL
# LM: Pre-trained Mistral 7B | Embeddings: Fine-tuned Mistral 7B
# Task: Combined LM+1NN with PERMANENT AUTHOR REMOVAL
# =============================================================================

set -e

# Create logs directory
mkdir -p logs

# GPU setup
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

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

# Fine-tuned model path
ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_weird_name_perturbed_20250825_225259"
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152305"

# Data files
TRAIN_FILE="dataset/private/tofu/tofu_train.json"
TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"

# KNN-LM parameters  
K=1
BATCH_SIZE=256

# DYNAMIC LAMBDA PARAMETERS
UPPER_LAMBDA=1.0
LOWER_LAMBDA=0.0
DISTANCE_THRESHOLD=0.4

# 🔥 NEW: AUTHOR REMOVAL PERCENTAGE (5%, 10%, 20%)
# Change this to test different removal percentages
# REMOVED_ENTITY_PERCENTAGE=0.05  # 5%
# REMOVED_ENTITY_PERCENTAGE=0.10  # 10%
REMOVED_ENTITY_PERCENTAGE=0.20  # 20%

# Generate percentage string for file naming
REMOVAL_PCT=$(python -c "import math; print(int($REMOVED_ENTITY_PERCENTAGE * 100))")

# Output files - Updated to include removal percentage in naming
REMOVED_OUTPUT_FILE="results/post_removal_accuracy_on_removed_and_retained/PI_perturbation_embedding/author_removal_${REMOVAL_PCT}pct_removed_authors_results.json"
# REMOVED_OUTPUT_FILE="results/post_removal_accuracy_on_removed_and_retained/name_perturbation_embedding/author_removal_${REMOVAL_PCT}pct_removed_authors_results.json"
RETAINED_OUTPUT_FILE="results/post_removal_accuracy_on_removed_and_retained/PI_perturbation_embedding/author_removal_${REMOVAL_PCT}pct_retained_authors_results.json"
# RETAINED_OUTPUT_FILE="results/post_removal_accuracy_on_removed_and_retained/name_perturbation_embedding/author_removal_${REMOVAL_PCT}pct_retained_authors_results.json"

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 KNN-LM with PERMANENT AUTHOR REMOVAL Configuration:"
echo "   Fine-tuned adapter: $ADAPTER_PATH"
echo "   Train file: $TRAIN_FILE"
echo "   Test file: $TEST_FILE"
echo "   Removed authors output: $REMOVED_OUTPUT_FILE"
echo "   Retained authors output: $RETAINED_OUTPUT_FILE"
echo "   K neighbors: $K"
echo "   Batch size: $BATCH_SIZE"
echo "   🔥 DYNAMIC LAMBDA SETTINGS:"
echo "   Upper lambda (near): $UPPER_LAMBDA"
echo "   Lower lambda (far): $LOWER_LAMBDA"
echo "   Distance threshold: $DISTANCE_THRESHOLD"
echo "   🗑️ AUTHOR REMOVAL SETTINGS:"
echo "   Removal percentage: ${REMOVAL_PCT}%"
echo "   🎯 TASK: Combined LM + 1NN with PERMANENT AUTHOR REMOVAL"
echo "   🤖 LM: Pre-trained Mistral 7B"
echo "   🎯 Embeddings: Fine-tuned Mistral 7B"

# Check if fine-tuned model exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ Fine-tuned adapter not found: $ADAPTER_PATH"
    exit 1
fi

if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
    echo "❌ Missing adapter_config.json in: $ADAPTER_PATH"
    exit 1
fi

if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ] && [ ! -f "$ADAPTER_PATH/adapter_model.bin" ]; then
    echo "❌ Missing adapter model files in: $ADAPTER_PATH"
    exit 1
fi

echo "✅ Fine-tuned adapter found and validated"

# Check if data files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Train file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

# Create output directories
mkdir -p "$(dirname "$REMOVED_OUTPUT_FILE")"
mkdir -p "$(dirname "$RETAINED_OUTPUT_FILE")"

# Validate removal percentage (using Python for float comparison)
VALID_PERCENTAGE=$(python -c "
percentage = $REMOVED_ENTITY_PERCENTAGE
if 0.0 <= percentage <= 1.0:
    print('valid')
else:
    print('invalid')
")

if [ "$VALID_PERCENTAGE" != "valid" ]; then
    echo "❌ Invalid removal percentage: $REMOVED_ENTITY_PERCENTAGE (should be 0.0-1.0)"
    exit 1
fi

# Validate other parameters
if [ $K -lt 1 ] || [ $K -gt 100 ]; then
    echo "❌ Invalid K value: $K (should be 1-100)"
    exit 1
fi

echo "✅ All parameters validated"

# =============================================================================
# RUN KNN-LM PIPELINE WITH PERMANENT AUTHOR REMOVAL
# =============================================================================

echo "🚀 Starting KNN-LM Pipeline with PERMANENT AUTHOR REMOVAL..."
echo "⏰ Processing ${REMOVAL_PCT}% author removal..."

python src/post_removal_accuracy_on_removed_and_retained/post_removal_accuracy_on_removed_and_retained.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --removed-output-file "$REMOVED_OUTPUT_FILE" \
    --retained-output-file "$RETAINED_OUTPUT_FILE" \
    --adapter-path "$ADAPTER_PATH" \
    --k $K \
    --upper-lambda $UPPER_LAMBDA \
    --lower-lambda $LOWER_LAMBDA \
    --distance-threshold $DISTANCE_THRESHOLD \
    --batch-size $BATCH_SIZE \
    --removed-entity-percentage $REMOVED_ENTITY_PERCENTAGE

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ KNN-LM with PERMANENT AUTHOR REMOVAL completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Permanent author removal: ${REMOVAL_PCT}% of authors"
    echo "   🤖 LM Model: Pre-trained Mistral 7B"
    echo "   🎯 Embedding Model: Fine-tuned Mistral 7B ($ADAPTER_PATH)"
    echo "   🗑️ Removed authors results: $REMOVED_OUTPUT_FILE"
    echo "   ✅ Retained authors results: $RETAINED_OUTPUT_FILE"
    echo "   🔥 Dynamic λ rule:"
    echo "     - Distance < $DISTANCE_THRESHOLD → λ = $UPPER_LAMBDA"
    echo "     - Distance >= $DISTANCE_THRESHOLD → λ = $LOWER_LAMBDA"
    
    # Show some stats if output files exist
    if [ -f "$REMOVED_OUTPUT_FILE" ]; then
        echo "📊 Removed authors file stats:"
        echo "   Size: $(du -h "$REMOVED_OUTPUT_FILE" | cut -f1)"
        REMOVED_COUNT=$(python -c "import json; print(len(json.load(open('$REMOVED_OUTPUT_FILE'))))" 2>/dev/null || echo "Unknown")
        echo "   Questions: $REMOVED_COUNT"
    fi
    
    if [ -f "$RETAINED_OUTPUT_FILE" ]; then
        echo "📊 Retained authors file stats:"
        echo "   Size: $(du -h "$RETAINED_OUTPUT_FILE" | cut -f1)"
        RETAINED_COUNT=$(python -c "import json; print(len(json.load(open('$RETAINED_OUTPUT_FILE'))))" 2>/dev/null || echo "Unknown")
        echo "   Questions: $RETAINED_COUNT"
    fi
    
    echo ""
    echo "🎯 QUICK GUIDE FOR DIFFERENT REMOVAL PERCENTAGES:"
    echo "   To test 5% removal:  REMOVED_ENTITY_PERCENTAGE=0.05"
    echo "   To test 10% removal: REMOVED_ENTITY_PERCENTAGE=0.10"
    echo "   To test 20% removal: REMOVED_ENTITY_PERCENTAGE=0.20"
    echo ""
    echo "📁 Output files will be named automatically:"
    echo "   5%:  author_removal_5pct_removed_authors_results.json"
    echo "   10%: author_removal_10pct_removed_authors_results.json"
    echo "   20%: author_removal_20pct_removed_authors_results.json"
    
else
    echo "❌ KNN-LM with PERMANENT AUTHOR REMOVAL failed!"
    exit 1
fi

echo "🎯 PERMANENT AUTHOR REMOVAL pipeline complete!"