#!/bin/bash

# =============================================================================
# CONFIGURATION: Dataset Selection for LoRA Fine-tuning
# =============================================================================

# --- TRAINING DATASET SELECTION (uncomment ONE) ---
# Option 1: Original private dataset (WITHOUT privacy protection)
DATA_FILE="dataset/private/tofu/tofu_train.json"
OUTPUT_DIR="./model_checkpoints/mistral_tofu_lora_tuned_$(date +%Y%m%d_%H%M%S)"

# Option 2: PI (Private Information) perturbation
# DATA_FILE="dataset/private/tofu/tofu_train_weird_name.json"
# OUTPUT_DIR="./model_checkpoints/mistral_tofu_lora_tuned_weird_name_perturbed_$(date +%Y%m%d_%H%M%S)"

# Option 3: Name perturbation
# NOTE: Multiple epsilon values available (ε = 0.5, 1, 2, 5, 8, 10)
#       Adjust the filename to match your desired epsilon value
# DATA_FILE="dataset/private/tofu/tofu_train_perturbed_mistral_corrected_eps10_0.json"
# OUTPUT_DIR="./model_checkpoints/mistral_tofu_lora_tuned_$(date +%Y%m%d_%H%M%S)"

# Option 4: DDPM (Deidentification via DP Masking)
# DATA_FILE="dataset/private/tofu/tofu_train_clustered_minimum_size_8_dp_20250826_221258.json"
# OUTPUT_DIR="./model_checkpoints/mistral_tofu_lora_tuned_name_perturbed_clustering_minimum_size_8_DP_$(date +%Y%m%d_%H%M%S)"

# NOTE: This script trains LoRA adapters on the PRIVATE dataset with different
#       privacy-preserving preprocessing methods applied to the data itself.
#       For DP-SGD training (privacy during training), use entity_level_DPSGD_lora_efficient.sh instead.
set -e

echo "🚀 Mistral 7B LoRA Fine-tuning Started: $(date)"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false # Prevents DataLoader forking issues with tokenizers
export HF_HOME="$HOME/.cache/huggingface"

# =============================================================================
# CONFIGURATION - STABLE HYPERPARAMETERS FOR TUNING
# =============================================================================

MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

NUM_EPOCHS=10 


BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=2e-4
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

MAX_LENGTH=512
NUM_EXAMPLES= # Use all examples by default. Uncomment and set if needed, e.g., NUM_EXAMPLES=5000

USE_4BIT=true
USE_8BIT=false
USE_FP16=true
USE_BF16=false

LOGGING_STEPS=50
SAVE_STEPS=9999999999999
SAVE_TOTAL_LIMIT=0

echo "🔧 Configuration for Fine-tuning:"
echo "   Model: $MODEL_NAME"
echo "   Data File: $DATA_FILE"
echo "   LoRA: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "   Training: epochs=$NUM_EPOCHS, batch_size=$BATCH_SIZE, grad_accum=$GRADIENT_ACCUMULATION"
echo "   Learning Rate: $LEARNING_RATE, Max Grad Norm: $MAX_GRAD_NORM"
echo "   Max Length: $MAX_LENGTH"
echo "   Num Examples: ${NUM_EXAMPLES:-All}"
echo "   Quantization: 4-bit=$USE_4BIT, 8-bit=$USE_8BIT"
echo "   Precision: FP16=$USE_FP16, BF16=$USE_BF16"
echo "   Output Dir: $OUTPUT_DIR"

# =============================================================================
# PRE-RUN CHECKS
# =============================================================================

echo "🔍 Performing pre-run checks..."

# Corrected: Check for lora_finetune_fixed.py
if [ ! -f "src/lora_finetune/lora_finetune.py" ]; then
    echo "❌ Error: Missing src/lora_finetune/lora_finetune.py. Ensure the script is in the current directory."
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Dataset file not found at $DATA_FILE. Please verify the path."
    exit 1
fi

mkdir -p logs
echo "✅ Checks passed. Proceeding with training."

# =============================================================================
# BUILD THE TRAINING COMMAND
# =============================================================================

# Corrected: Reference lora_finetune_fixed.py
PYTHON_SCRIPT="src/lora_finetune/lora_finetune.py"

CMD_ARGS="--model-name $MODEL_NAME"
CMD_ARGS="$CMD_ARGS --data-file $DATA_FILE"
CMD_ARGS="$CMD_ARGS --output-dir $OUTPUT_DIR"
CMD_ARGS="$CMD_ARGS --lora-r $LORA_R"
CMD_ARGS="$CMD_ARGS --lora-alpha $LORA_ALPHA"
CMD_ARGS="$CMD_ARGS --lora-dropout $LORA_DROPOUT"
CMD_ARGS="$CMD_ARGS --num-epochs $NUM_EPOCHS"
CMD_ARGS="$CMD_ARGS --batch-size $BATCH_SIZE"
CMD_ARGS="$CMD_ARGS --gradient-accumulation-steps $GRADIENT_ACCUMULATION"
CMD_ARGS="$CMD_ARGS --learning-rate $LEARNING_RATE"
CMD_ARGS="$CMD_ARGS --warmup-ratio $WARMUP_RATIO"
CMD_ARGS="$CMD_ARGS --weight-decay $WEIGHT_DECAY"
CMD_ARGS="$CMD_ARGS --max-grad-norm $MAX_GRAD_NORM"
CMD_ARGS="$CMD_ARGS --max-length $MAX_LENGTH"
CMD_ARGS="$CMD_ARGS --logging-steps $LOGGING_STEPS"
CMD_ARGS="$CMD_ARGS --save-steps $SAVE_STEPS"
CMD_ARGS="$CMD_ARGS --save-total-limit $SAVE_TOTAL_LIMIT"

if [ -n "$NUM_EXAMPLES" ]; then
    CMD_ARGS="$CMD_ARGS --num-examples $NUM_EXAMPLES"
fi

if [ "$USE_4BIT" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --use-4bit"
fi

if [ "$USE_8BIT" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --use-8bit"
fi

if [ "$USE_FP16" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --fp16"
fi

if [ "$USE_BF16" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --bf16"
fi

echo "Generated Command: python $PYTHON_SCRIPT $CMD_ARGS"

# =============================================================================
# RUN TRAINING
# =============================================================================

echo ""
echo "🚀 Starting training run..."
echo "============================================================"

python -u "$PYTHON_SCRIPT" $CMD_ARGS

exit_code=$?

echo "============================================================"
echo "📅 Training completed at: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ Training completed successfully!"

    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "📁 Contents of output directory '$OUTPUT_DIR':"
        ls -la "$OUTPUT_DIR"

        if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ] || [ -f "$OUTPUT_DIR/adapter_model.bin" ]; then
            echo "✅ LoRA adapter model saved successfully."
        else
            echo "⚠️ Warning: LoRA adapter model file (adapter_model.safetensors or .bin) not found."
        fi
        if [ -f "$OUTPUT_DIR/tokenizer.json" ]; then
            echo "✅ Tokenizer files saved successfully."
        else
            echo "⚠️ Warning: Tokenizer files not found."
        fi
    else
        echo "❌ Error: Output directory '$OUTPUT_DIR' was not created or is empty."
    fi

    echo ""
    echo "🎯 To test the fine-tuned model, run the following Python snippet (replace with your desired prompt):"
    echo "------------------------------------------------------------"
    echo "python -c \""
    echo "from transformers import AutoTokenizer, AutoModelForCausalLM"
    echo "from peft import PeftModel"
    echo "import torch"
    echo ""
    echo "model_name = '$MODEL_NAME'"
    echo "tuned_model_dir = '$OUTPUT_DIR'"
    echo ""
    echo "print(f'Loading tokenizer from {model_name}')"
    echo "tokenizer = AutoTokenizer.from_pretrained(model_name)"
    echo "if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token"
    echo ""
    echo "print(f'Loading base model {model_name} in float16 (or device_map=auto if memory allows)... ') "
    echo "base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')"
    echo ""
    echo "print(f'Loading LoRA adapter from {tuned_model_dir}...') "
    echo "model = PeftModel.from_pretrained(base_model, tuned_model_dir)"
    echo "model.eval()"
    echo ""
    echo "prompt = '<s>[INST] What is machine learning? [/INST]'"
    echo "print(f'\\nGenerating for prompt: {prompt}')"
    echo "inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=True).to(model.device)"
    echo ""
    echo "with torch.no_grad():"
    echo "    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)"
    echo "generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)"
    echo "print(f'\\nGenerated Output (with special tokens):\\n{generated_text}')"
    echo "print(f'\\nGenerated Output (skipped special tokens):\\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}') "
    echo "\""
    echo "------------------------------------------------------------"

else
    echo "❌ Training failed with exit code: $exit_code"
fi

echo ""
echo "🏁 Script Finished: $(date)"

exit $exit_code