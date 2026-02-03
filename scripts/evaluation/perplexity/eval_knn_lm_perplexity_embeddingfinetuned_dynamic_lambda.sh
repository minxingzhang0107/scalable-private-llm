#!/bin/bash

set -e

# Create logs directory
mkdir -p logs

# =============================================================================
# CONFIGURATION: Model and Dataset Selection
# =============================================================================

# --- EMBEDDING FUNCTION SELECTION (uncomment ONE) ---
# Option 1: Trained on PUBLIC + PRIVATE without privacy protection 
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152332"

# Option 2: Trained with PUBLIC + PRIVATE with PI (Private Information) perturbation
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_weird_name_perturbed_20250825_225259"

# Option 3: Trained with PUBLIC + PRIVATE with DDPM (Deidentification by DP Masking)
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_name_perturbed_clustering_minimum_size_8_DP_20250826_222158"

# Option 4: Trained with PUBLIC + PRIVATE with name perturbation (ε = 0.5)
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152052"

# Option 5: Trained with PUBLIC + PRIVATE with name perturbation (ε = 1)
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152118"

# Option 6: Trained with PUBLIC + PRIVATE with name perturbation (ε = 2)
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152143"

# Option 7: Trained with PUBLIC + PRIVATE with name perturbation (ε = 5)
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152200"

# Option 8: Trained with PUBLIC + PRIVATE with name perturbation (ε = 8)
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152220"

# Option 9: Trained with PUBLIC + PRIVATE with name perturbation (ε = 10)
# ADAPTER_PATH="./model_checkpoints/mistral_tofu_lora_tuned_20250804_152305"

# Option 10: Trained with PUBLIC + PRIVATE with DP-SGD 
ADAPTER_PATH="./model_checkpoints/user_dp_lora_mistral_20251003_203637"

# --- DATASET SELECTION ---
# Train dataset
TRAIN_FILE="dataset/private/tofu/tofu_train.json"

# Test dataset (uncomment ONE)
TEST_FILE="dataset/private/tofu/tofu_test_question_paraphrased.json"  # Private
# TEST_FILE="dataset/public/public_test_tiny_qa.json"  # Public


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
K=1                         # Number of neighbors for KNN (used for building datastore)
BATCH_SIZE=256             # Batch size for A6000

# 🔥 DYNAMIC LAMBDA PARAMETERS
UPPER_LAMBDA=1.0           # High KNN weight when distance < threshold (close neighbors)
LOWER_LAMBDA=0.0           # Low KNN weight when distance >= threshold (far neighbors)  
DISTANCE_THRESHOLD=0.4     # Distance threshold for lambda assignment

# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

echo "🔧 KNN-LM Perplexity Evaluation with Fine-tuned Embeddings Configuration:"
echo "   Fine-tuned adapter: $ADAPTER_PATH"
echo "   Train file: $TRAIN_FILE"
echo "   Test file: $TEST_FILE"
echo "   K neighbors: $K"
echo "   Batch size: $BATCH_SIZE"
echo "   🔥 DYNAMIC LAMBDA SETTINGS:"
echo "   Upper lambda (near): $UPPER_LAMBDA"
echo "   Lower lambda (far): $LOWER_LAMBDA"
echo "   Distance threshold: $DISTANCE_THRESHOLD"
echo "   🎯 TASK: KNN-LM Perplexity Evaluation with Combined LM + 1NN (DYNAMIC λ) + FINE-TUNED EMBEDDINGS"
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
# RUN KNN-LM PERPLEXITY EVALUATION WITH FINE-TUNED EMBEDDINGS
# =============================================================================

echo "🚀 Starting KNN-LM Perplexity Evaluation Pipeline with FINE-TUNED EMBEDDINGS..."
echo "⏰ NO TIME LIMIT - Will run until completion!"

python src/evaluation/perplexity/eval_knn_lm_perplexity_embeddingfinetuned_dynamic_lambda.py \
    --train-file "$TRAIN_FILE" \
    --test-file "$TEST_FILE" \
    --adapter-path "$ADAPTER_PATH" \
    --k $K \
    --upper-lambda $UPPER_LAMBDA \
    --lower-lambda $LOWER_LAMBDA \
    --distance-threshold $DISTANCE_THRESHOLD \
    --batch-size $BATCH_SIZE

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ KNN-LM Perplexity Evaluation with FINE-TUNED EMBEDDINGS completed successfully!"
    echo "🎯 Results Summary:"
    echo "   ✅ KNN-LM Perplexity with Combined LM+1NN (DYNAMIC λ) + FINE-TUNED EMBEDDINGS calculated"
    echo "   🤖 LM Model: Pre-trained Mistral 7B"
    echo "   🎯 Embedding Model: Fine-tuned Mistral 7B ($ADAPTER_PATH)"
    echo "   🔥 Dynamic λ rule:"
    echo "     - Distance < $DISTANCE_THRESHOLD → λ = $UPPER_LAMBDA (high KNN weight)"
    echo "     - Distance >= $DISTANCE_THRESHOLD → λ = $LOWER_LAMBDA (low KNN weight)"
    echo "   📊 See perplexity results above"
else
    echo "❌ KNN-LM Perplexity Evaluation with FINE-TUNED EMBEDDINGS failed!"
    exit 1
fi

echo "🎯 Fine-tuned embedding perplexity evaluation pipeline complete!"