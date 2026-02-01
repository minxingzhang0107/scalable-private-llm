#!/usr/bin/bash

# =============================================================================
# KNN-LM Time Analysis: Test different k values
# Analyzes: Total time, FAISS search time, Distribution computation time
# =============================================================================

set -e

# Create logs directory
mkdir -p logs


# K values to test
# K_VALUES=(1 3 5 7 9)
K_VALUES=(1 3 5 7 9 10 15 20 30 50 100 500)

# Loop through each k value
for CURRENT_K in "${K_VALUES[@]}"; do

echo "🔢 Testing k = $CURRENT_K"

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
"

# =============================================================================
# PARAMETERS
# =============================================================================

ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152332"
TRAIN_FILE="dataset/private/tofu/tofu_train.json"
TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"

BATCH_SIZE=256
UPPER_LAMBDA=1.0
LOWER_LAMBDA=0.0
DISTANCE_THRESHOLD=0.4

# Output file with k value in name
OUTPUT_FILE="results/analysis/k_${CURRENT_K}/time_analysis_k_${CURRENT_K}.json"

echo "🔧 Time Analysis Configuration:"
echo "   k = $CURRENT_K"
echo "   Train file: $TRAIN_FILE"
echo "   Test file: $TEST_FILE"
echo "   Output: $OUTPUT_FILE"
echo "   Adapter: $ADAPTER_PATH"

# Validate files
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Train file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ Adapter not found: $ADAPTER_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "✅ All parameters validated"

# =============================================================================
# RUN TIME ANALYSIS
# =============================================================================

echo "🚀 Starting time analysis for k=$CURRENT_K..."

python src/plot/generation_time_vs_k.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --output-file "$OUTPUT_FILE" \
    --adapter-path "$ADAPTER_PATH" \
    --k $CURRENT_K \
    --upper-lambda $UPPER_LAMBDA \
    --lower-lambda $LOWER_LAMBDA \
    --distance-threshold $DISTANCE_THRESHOLD \
    --batch-size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo "✅ Time analysis for k=$CURRENT_K completed successfully!"
    echo "💾 Results saved to: $OUTPUT_FILE"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "📊 Output file stats:"
        echo "   Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    fi
else
    echo "❌ Time analysis for k=$CURRENT_K failed!"
    exit 1
fi

echo "🎯 Time analysis for k=$CURRENT_K complete!"

done