#!/bin/bash

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
AUTHOR_MAPPING_PATH="dataset/private/tofu/author_names_mapping.json"

# --- Output file paths ---
OUTPUT_EMBEDDINGS="dataset/private/tofu/author_embeddings.npy"
OUTPUT_NAMES="dataset/private/tofu/author_names_ordered.json"

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

python_script="src/deidentification_by_DP_masking/compute_name_embedding.py"

echo "🚀 Starting Standalone Embedding Generation Script"
echo "====================================================="
echo "   - Model: $MODEL_PATH"
echo "   - Author Source: $AUTHOR_MAPPING_PATH"
echo "   - Output Embeddings: $OUTPUT_EMBEDDINGS"
echo "   - Output Names: $OUTPUT_NAMES"
echo "====================================================="

# Create directories
mkdir -p logs

# Check for required files
if [ ! -f "$python_script" ]; then
    echo "❌ Python script '$python_script' not found!"
    exit 1
fi
if [ ! -f "src/name_perturbation/dp_author_name_perturbation.py" ]; then
    echo "❌ Helper script 'dp_author_name_perturbation.py' not found in src/name_perturbation directory!"
    exit 1
fi
if [ ! -f "$AUTHOR_MAPPING_PATH" ]; then
    echo "❌ Author mapping file not found at '$AUTHOR_MAPPING_PATH'!"
    exit 1
fi

# Build command line arguments
PYTHON_ARGS=(
    --model-path "$MODEL_PATH"
    --author-mapping-path "$AUTHOR_MAPPING_PATH"
    --output-embeddings-path "$OUTPUT_EMBEDDINGS"
    --output-names-path "$OUTPUT_NAMES"
)

# Run the Python script
echo "🧠 Executing Python script to generate embeddings..."
python -u "$python_script" "${PYTHON_ARGS[@]}"

exit_code=${PIPESTATUS[0]}

echo "====================================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ Embedding generation completed successfully!"
    echo "   Files created:"
    echo "   - $(du -h $OUTPUT_EMBEDDINGS)"
    echo "   - $(du -h $OUTPUT_NAMES)"
else
    echo "❌ Script failed with exit code: $exit_code"
fi