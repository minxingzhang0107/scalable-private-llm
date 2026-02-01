#!/bin/bash

set -e

# =============================================================================
# Author Name Clustering with Differential Privacy
# =============================================================================

echo "🚀 De-identification by DP masking Started: $(date)"
echo "✅ This script clusters author names and applies differential privacy."

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HOME/.cache/huggingface"

# Clear GPU memory before starting
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('🧹 GPU memory cleared')
"

# =============================================================================
# MODEL AND DATA CONFIGURATION
# =============================================================================

MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
AUTHOR_MAPPING_FILE="dataset/private/tofu/author_names_mapping.json"  # Same as reference
TRAIN_FILE="dataset/private/tofu/tofu_train.json"  # Same as reference
OUTPUT_DIR="dataset/private/tofu"
OUTPUT_FILE="$OUTPUT_DIR/tofu_train_clustered_minimum_size_8_dp_$(date +%Y%m%d_%H%M%S).json"

# =============================================================================
# CLUSTERING AND PRIVACY PARAMETERS
# You can modify these values or pass them as command line arguments
# =============================================================================

# Default values - modify these as needed
MINIMUM_CLUSTER_SIZE=8      # Minimum number of authors per cluster
CLIP_THRESHOLD=352.0         # C: L2 norm clipping threshold for embeddings
EPSILON=5.0                 # ε: Privacy parameter (smaller = more private)
DELTA=1e-5                  # δ: Privacy parameter (smaller = more private)

# Check for command line arguments to override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --minimum-cluster-size)
            MINIMUM_CLUSTER_SIZE="$2"
            shift 2
            ;;
        --clip-threshold)
            CLIP_THRESHOLD="$2"
            shift 2
            ;;
        --epsilon)
            EPSILON="$2"
            shift 2
            ;;
        --delta)
            DELTA="$2"
            shift 2
            ;;
        --author-mapping-file)
            AUTHOR_MAPPING_FILE="$2"
            shift 2
            ;;
        --train-file)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --minimum-cluster-size NUM   Minimum number of authors per cluster (default: $MINIMUM_CLUSTER_SIZE)"
            echo "  --clip-threshold FLOAT       L2 norm clipping threshold (default: $CLIP_THRESHOLD)"
            echo "  --epsilon FLOAT              Privacy parameter epsilon (default: $EPSILON)"
            echo "  --delta FLOAT                Privacy parameter delta (default: $DELTA)"
            echo "  --author-mapping-file PATH   Path to author mapping JSON file (default: $AUTHOR_MAPPING_FILE)"
            echo "  --train-file PATH            Path to training data JSON file (default: $TRAIN_FILE)"
            echo "  --output-file PATH           Output file path (default: auto-generated)"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --minimum-cluster-size 5 --clip-threshold 8.0 --epsilon 0.5 --delta 1e-6"
            echo "  $0 --epsilon 2.0 --delta 1e-4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# PRE-RUN VALIDATION
# =============================================================================

PYTHON_SCRIPT="src/deidentification_by_DP_masking/deidentification_by_DP_masking.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Missing Python script: $PYTHON_SCRIPT"
    echo "   Make sure the Python file exists at: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -f "$AUTHOR_MAPPING_FILE" ]; then
    echo "❌ Error: Author mapping file not found at $AUTHOR_MAPPING_FILE"
    echo "   Please check the path or use --author-mapping-file to specify correct location"
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Error: Training data file not found at $TRAIN_FILE"
    echo "   Please check the path or use --train-file to specify correct location"
    exit 1
fi

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Validate parameters
if (( $(echo "$MINIMUM_CLUSTER_SIZE < 1" | bc -l) )); then
    echo "❌ Error: minimum-cluster-size must be >= 1"
    exit 1
fi

if (( $(echo "$CLIP_THRESHOLD <= 0" | bc -l) )); then
    echo "❌ Error: clip-threshold must be > 0"
    exit 1
fi

if (( $(echo "$EPSILON <= 0" | bc -l) )); then
    echo "❌ Error: epsilon must be > 0"
    exit 1
fi

if (( $(echo "$DELTA <= 0 || $DELTA >= 1" | bc -l) )); then
    echo "❌ Error: delta must be in (0, 1)"
    exit 1
fi

# Check system
echo "🧮 System check..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

echo "✅ Pre-run checks completed."

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

echo "✅ Configuration Summary:"
echo "   Model: $MODEL_NAME"
echo "   Author Mapping File: $AUTHOR_MAPPING_FILE"
echo "   Training Data File: $TRAIN_FILE"
echo "   Output File: $OUTPUT_FILE"
echo ""
echo "   📊 Clustering Parameters:"
echo "      Minimum Cluster Size: $MINIMUM_CLUSTER_SIZE"
echo "      Clipping Threshold (C): $CLIP_THRESHOLD"
echo ""
echo "   🔒 Privacy Parameters:"
echo "      Epsilon (ε): $EPSILON"
echo "      Delta (δ):   $DELTA"

# Calculate expected noise level for information
EXPECTED_NOISE=$(python -c "
import math
sigma = $CLIP_THRESHOLD * math.sqrt(2 * math.log(1.25 / $DELTA)) / $EPSILON
print(f'{sigma:.6f}')
")
echo "      Expected Noise Std: $EXPECTED_NOISE"

# =============================================================================
# BUILD COMMAND AND RUN
# =============================================================================

CMD_ARGS="--model-name $MODEL_NAME"
CMD_ARGS="$CMD_ARGS --author-mapping-file $AUTHOR_MAPPING_FILE"
CMD_ARGS="$CMD_ARGS --train-file $TRAIN_FILE"
CMD_ARGS="$CMD_ARGS --output-file $OUTPUT_FILE"
CMD_ARGS="$CMD_ARGS --minimum-cluster-size $MINIMUM_CLUSTER_SIZE"
CMD_ARGS="$CMD_ARGS --clip-threshold $CLIP_THRESHOLD"
CMD_ARGS="$CMD_ARGS --epsilon $EPSILON"
CMD_ARGS="$CMD_ARGS --delta $DELTA"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "🚀 Starting De-identification by DP masking..."
echo "Command: python -u $PYTHON_SCRIPT $CMD_ARGS"
echo "============================================================"

start_time=$(date +%s)

# Execute the command and log output
python -u "$PYTHON_SCRIPT" $CMD_ARGS 2>&1 | tee -a "logs/deidentification_by_DP_masking_${TIMESTAMP}.log"
exit_code=${PIPESTATUS[0]}

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "============================================================"
echo "📅 Clustering completed at: $(date)"
echo "⏱️ Duration: ${duration}s ($(python -c "print(f'{$duration/60:.1f} min')"))"

if [ $exit_code -eq 0 ]; then
    echo "✅ De-identification by DP masking completed successfully!"
    echo "📁 Updated training data saved to $OUTPUT_FILE"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "📊 File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        
        # Show brief preview of results
        echo "🔍 Results preview:"
        python -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    print(f'   Total training examples: {len(data)}')
    
    # Count author distribution
    from collections import defaultdict
    author_counts = defaultdict(int)
    for example in data:
        if 'author' in example:
            author_counts[example['author']] += 1
    
    print(f'   Unique authors in updated data: {len(author_counts)}')
    print('   Author distribution (top 5):')
    sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
    for author, count in sorted_authors[:5]:
        print(f'      {author}: {count} examples')
    if len(sorted_authors) > 5:
        print(f'      ... and {len(sorted_authors)-5} more authors')
        
except Exception as e:
    print(f'   Error reading results: {e}')
"
    else
        echo "⚠️ Warning: Output file not found"
    fi
else
    echo "❌ De-identification by DP masking failed with exit code: $exit_code"
    echo "🔧 TROUBLESHOOTING:"
    echo "   Check log file for specific errors:"
    echo "   - Full Log: logs/deidentification_by_DP_masking_${TIMESTAMP}.log"
    echo ""
    echo "   Common issues:"
    echo "   - Author names mapping file not found or invalid format"
    echo "   - Training data file not found or invalid format"
    echo "   - Insufficient GPU memory for model loading"
    echo "   - Invalid parameter values"
    echo "   - Minimum cluster size too large for the number of authors"
fi

echo ""
echo "🏁 Script Finished: $(date)"

exit $exit_code