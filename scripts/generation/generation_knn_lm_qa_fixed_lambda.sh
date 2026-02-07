#!/bin/bash

# =============================================================================
# CONFIGURATION: Dataset, Lambda, and Output Selection
# =============================================================================

# --- TRAIN DATASET ---
TRAIN_FILE="dataset/private/tofu/tofu_train.json"

# --- TEST DATASET SELECTION (uncomment ONE) ---
# Option A: Test on PRIVATE data
# TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"

# Option B: Test on PUBLIC data
TEST_FILE="dataset/public/public_test_tiny_qa.json"

# --- KNN-LM PARAMETERS ---
LAMBDA_WEIGHT=0.75     # Weight for KNN vs LM (options: 0.25, 0.5, 0.75)

# =============================================================================
# IMPORTANT: Output Directory Configuration
# =============================================================================
# You MUST also modify the Python script (lines 803-805) to match your configuration:
#
# For PRIVATE data:
#   output_dir = "results/private/tofu/fixed_lambda_pretrained_embedding"
#   output_file = os.path.join(output_dir, "fixed_lambda_pretrained_embedding_lambda_0_75.json")
#
# For PUBLIC data:
#   output_dir = "results/public/fixed_lambda_pretrained_embedding"
#   output_file = os.path.join(output_dir, "fixed_lambda_pretrained_embedding_lambda_0_75.json")
#
# Update the lambda value in filename to match LAMBDA_WEIGHT above:
#   - LAMBDA_WEIGHT=0.25 → "fixed_lambda_pretrained_embedding_lambda_0_25.json"
#   - LAMBDA_WEIGHT=0.5  → "fixed_lambda_pretrained_embedding_lambda_0_5.json"
#   - LAMBDA_WEIGHT=0.75 → "fixed_lambda_pretrained_embedding_lambda_0_75.json"
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
# KNN-LM parameters  
K=1                    # Number of neighbors for KNN (used in tasks 1c and 2a)
BATCH_SIZE=256         # Batch size for A6000

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 Five-Task KNN-LM Configuration:"
echo "   Train file: $TRAIN_FILE"
echo "   Test file: $TEST_FILE"
echo "   K neighbors: $K"
echo "   Lambda weight: $LAMBDA_WEIGHT"
echo "   Batch size: $BATCH_SIZE"
echo "   🎯 FIVE TASKS:"
echo "     1a: LM-only generation"
echo "     1b: 1NN-only generation"
echo "     1c: KNN-only generation (k=$K neighbors)"
echo "     2a: Combined LM + KNN (λ=$LAMBDA_WEIGHT)"
echo "     2b: Combined LM + 1NN (λ=$LAMBDA_WEIGHT)"

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

# Validate Lambda parameter
if (( $(echo "$LAMBDA_WEIGHT < 0.0" | bc -l) )) || (( $(echo "$LAMBDA_WEIGHT > 1.0" | bc -l) )); then
    echo "❌ Invalid Lambda weight: $LAMBDA_WEIGHT (should be 0.0-1.0)"
    exit 1
fi

echo "✅ All parameters validated"

# =============================================================================
# RUN FIVE-TASK KNN-LM GENERATION
# =============================================================================

echo "🚀 Starting Five-Task KNN-LM Generation Pipeline..."

python src/generation/generation_knn_lm_qa_fixed_lambda.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --k $K \
    --lambda-weight $LAMBDA_WEIGHT \
    --batch-size $BATCH_SIZE

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Five-Task KNN-LM generation completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Task 1a: LM-only generation completed"
    echo "   ✅ Task 1b: 1NN-only generation completed"
    echo "   ✅ Task 1c: KNN-only generation (k=$K) completed"
    echo "   ✅ Task 2a: Combined LM+KNN (λ=$LAMBDA_WEIGHT) completed"
    echo "   ✅ Task 2b: Combined LM+1NN (λ=$LAMBDA_WEIGHT) completed"
else
    echo "❌ Five-Task KNN-LM generation failed!"
    exit 1
fi

echo "🎯 Pipeline complete!"