#!/bin/bash

# =============================================================================
# Distance Histogram Analysis Pipeline
# Collect shortest distances during KNN-LM generation and plot histograms
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

# Fine-tuned model path - CHANGE THIS TO YOUR ACTUAL FINE-TUNED MODEL PATH
# normal finetuned
# ADAPTER_PATH="./mistral_tofu_lora_tuned_20250804_152332"

# finetuned with name perturbation
# ADAPTER_PATH="./mistral_tofu_lora_tuned_20250804_152052"
# ADAPTER_PATH="./mistral_tofu_lora_tuned_20250804_152118"
# ADAPTER_PATH="./mistral_tofu_lora_tuned_20250804_152143"
# ADAPTER_PATH="./mistral_tofu_lora_tuned_20250804_152200"
# ADAPTER_PATH="./mistral_tofu_lora_tuned_20250804_152220"
# ADAPTER_PATH="./mistral_tofu_lora_tuned_20250804_152305"

# finetuned with name perturbation using clustering dp
# ADAPTER_PATH="./mistral_tofu_lora_tuned_name_perturbed_clustering_minimum_size_8_DP_20250826_222158"

# finetuned with differential privacy 
# ADAPTER_PATH="./simple_lora_dp_mistral_20250804_181227"

# finetuned with weird word: toy example 
# ADAPTER_PATH="./mistral_tofu_lora_tuned_weird_name_raw_20250818_130356"
# ADAPTER_PATH="./mistral_tofu_lora_tuned_weird_name_perturbed_20250818_130856"

# finetuned with weird word
ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_weird_name_perturbed_20250825_225259"

# Data files
TRAIN_FILE="dataset/private/tofu/tofu_train.json"
PRIVATE_TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"
PUBLIC_TEST_FILE="dataset/public/public_test_tiny_qa.json"

# KNN-LM parameters  
K=1                         # Number of neighbors for KNN
BATCH_SIZE=256             # Batch size for A6000

# 🔥 DYNAMIC LAMBDA PARAMETERS
UPPER_LAMBDA=1.0           # High KNN weight when distance < threshold (close neighbors)
LOWER_LAMBDA=0.0           # Low KNN weight when distance >= threshold (far neighbors)  
DISTANCE_THRESHOLD=0.4     # Distance threshold for analysis

# Output plot file
OUTPUT_PLOT="results/analysis/distance_histogram_weird_word_perturbed_threshold_${DISTANCE_THRESHOLD}_range_0_1.png"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PLOT")"

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 Distance Histogram Analysis Configuration:"
echo "   Fine-tuned adapter: $ADAPTER_PATH"
echo "   Train file: $TRAIN_FILE"
echo "   Private test file: $PRIVATE_TEST_FILE"
echo "   Public test file: $PUBLIC_TEST_FILE"
echo "   Output plot: $OUTPUT_PLOT"
echo "   K neighbors: $K"
echo "   Batch size: $BATCH_SIZE"
echo "   Distance threshold: $DISTANCE_THRESHOLD"
echo "   🎯 TASK: Collect shortest distances and plot histogram"
echo "   🤖 LM: Pre-trained Mistral 7B"
echo "   🎯 Embeddings: Fine-tuned Mistral 7B"

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

# Check if files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Train file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$PRIVATE_TEST_FILE" ]; then
    echo "❌ Private test file not found: $PRIVATE_TEST_FILE"
    exit 1
fi

if [ ! -f "$PUBLIC_TEST_FILE" ]; then
    echo "❌ Public test file not found: $PUBLIC_TEST_FILE"
    exit 1
fi

# Validate K parameter
if [ $K -lt 1 ] || [ $K -gt 100 ]; then
    echo "❌ Invalid K value: $K (should be 1-100)"
    exit 1
fi

# Validate Distance Threshold
if (( $(echo "$DISTANCE_THRESHOLD < 0.0" | bc -l) )) || (( $(echo "$DISTANCE_THRESHOLD > 2.0" | bc -l) )); then
    echo "❌ Invalid Distance Threshold: $DISTANCE_THRESHOLD (should be 0.0-2.0)"
    exit 1
fi

echo "✅ All parameters validated"

# =============================================================================
# RUN DISTANCE HISTOGRAM ANALYSIS
# =============================================================================

echo "🚀 Starting Distance Histogram Analysis..."
echo "⏰ NO TIME LIMIT - Will run until completion!"

python src/plot/shortest_distance_histogram_plot.py \
    --train-file "$TRAIN_FILE" \
    --private-test-file "$PRIVATE_TEST_FILE" \
    --public-test-file "$PUBLIC_TEST_FILE" \
    --output-plot "$OUTPUT_PLOT" \
    --adapter-path "$ADAPTER_PATH" \
    --k $K \
    --upper-lambda $UPPER_LAMBDA \
    --lower-lambda $LOWER_LAMBDA \
    --distance-threshold $DISTANCE_THRESHOLD \
    --batch-size $BATCH_SIZE

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Distance Histogram Analysis completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ Shortest distances collected for both datasets"
    echo "   🤖 LM Model: Pre-trained Mistral 7B"
    echo "   🎯 Embedding Model: Fine-tuned Mistral 7B ($ADAPTER_PATH)"
    echo "   🔥 Dynamic λ rule:"
    echo "     - Distance < $DISTANCE_THRESHOLD → λ = $UPPER_LAMBDA (high KNN weight)"
    echo "     - Distance >= $DISTANCE_THRESHOLD → λ = $LOWER_LAMBDA (low KNN weight)"
    echo "   📈 Histogram saved to: $OUTPUT_PLOT"
    
    # Show some stats if output file exists
    if [ -f "$OUTPUT_PLOT" ]; then
        echo "📊 Output plot stats:"
        echo "   Size: $(du -h "$OUTPUT_PLOT" | cut -f1)"
        echo "   Format: PNG (with PDF version also saved)"
        
        # Also check if PDF exists
        PDF_FILE="${OUTPUT_PLOT%.png}.pdf"
        if [ -f "$PDF_FILE" ]; then
            echo "   PDF version: $PDF_FILE"
        fi
    fi
else
    echo "❌ Distance Histogram Analysis failed!"
    exit 1
fi

echo "🎯 Distance analysis pipeline complete!"