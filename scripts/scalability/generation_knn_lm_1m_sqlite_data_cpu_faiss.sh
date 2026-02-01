#!/bin/bash

set -e

mkdir -p logs
mkdir -p results/private/syn_traj

# GPU CONFIGURATION
# Users can modify NUM_GPUS based on their available hardware
NUM_GPUS=4  # Set to number of GPUs you want to use (1, 2, 4, or 8)

# Only set CUDA_VISIBLE_DEVICES if not already set by user/SLURM
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    if [ $NUM_GPUS -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
    elif [ $NUM_GPUS -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    elif [ $NUM_GPUS -eq 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [ $NUM_GPUS -eq 8 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    else
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
    fi
fi

echo "🖥️  GPU Configuration:"
echo "   Requested GPUs: $NUM_GPUS"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
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

# # PARAMETERS
# TRAIN_FILE="dataset/private/syn_traj/1m_facts_train.json"
# TEST_FILE="dataset/private/syn_traj/1m_facts_test.json"
# OUTPUT_FILE="results/private/syn_traj/knn_lm_answers_A5000.json"
# DATASTORE_DIR="/usr/xtmp/mz238/datastore_1m_A5000"

# PARAMETERS
TRAIN_FILE="dataset/private/syn_traj/scalability_dataset/train_100k.json"
TEST_FILE="dataset/private/syn_traj/scalability_dataset/test_100k.json"
OUTPUT_FILE="results/private/syn_traj/knn_lm_answers_100k_A5000.json"
DATASTORE_DIR="/usr/xtmp/mz238/datastore_100k_A5000"

# # PARAMETERS
# TRAIN_FILE="dataset/private/syn_traj/intermediate_train_datasets/train_1000k.json"
# TEST_FILE="dataset/private/syn_traj/scalability_dataset/test_1M.json"
# OUTPUT_FILE="results/private/syn_traj/knn_lm_answers_820k_A5000.json"
# DATASTORE_DIR="/usr/xtmp/mz238/datastore_820k_A5000"


# TRAIN_FILE="dataset/private/syn_traj/toy_example/5_facts_train.json"
# TEST_FILE="dataset/private/syn_traj/toy_example/5_facts_train.json"
# OUTPUT_FILE="results/private/syn_traj/knn_lm_answers_A5000_toy_example.json"
# DATASTORE_DIR="/usr/xtmp/mz238/datastore_1m_A5000_toy_example"

K=1
BATCH_SIZE=768
UPPER_LAMBDA=1.0
LOWER_LAMBDA=0.0
DISTANCE_THRESHOLD=0.4
USE_IVF=true
FORCE_REBUILD=false  # Set to true to rebuild even if datastore exists

echo ""
echo "🔧 CONFIGURATION:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🖥️  GPUs: $NUM_GPUS × A5000"
echo "📁 Train: $TRAIN_FILE"
echo "📁 Test: $TEST_FILE"
echo "💾 Output: $OUTPUT_FILE"
echo "💾 Datastore: $DATASTORE_DIR"
echo "🔢 Batch size: $BATCH_SIZE"
echo "🔢 k-NN: $K"
echo "⚡ IVF: $([ "$USE_IVF" = true ] && echo 'ENABLED' || echo 'DISABLED')"
echo "🔄 Force rebuild: $([ "$FORCE_REBUILD" = true ] && echo 'YES' || echo 'NO')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# VALIDATION
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Train file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

echo "✅ Data files validated"
mkdir -p "$(dirname "$OUTPUT_FILE")"

# RUN
echo ""
echo "🚀 STARTING PIPELINE..."
echo ""

CMD="python src/scalability/generation_knn_lm_1m_sqlite_data_cpu_faiss_detailed_time_analysis.py \
    --train-file \"$TRAIN_FILE\" \
    --test-file \"$TEST_FILE\" \
    --output-file \"$OUTPUT_FILE\" \
    --k $K \
    --upper-lambda $UPPER_LAMBDA \
    --lower-lambda $LOWER_LAMBDA \
    --distance-threshold $DISTANCE_THRESHOLD \
    --batch-size $BATCH_SIZE \
    --datastore-dir \"$DATASTORE_DIR\" \
    --num-gpus $NUM_GPUS"

if [ "$USE_IVF" = true ]; then
    CMD="$CMD --use-ivf"
fi

if [ "$FORCE_REBUILD" = true ]; then
    CMD="$CMD --force-rebuild"
fi

eval $CMD

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ PIPELINE COMPLETED"
    echo ""
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        NUM_QUESTIONS=$(python -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))" 2>/dev/null || echo "Unknown")
        echo "   Output: $OUTPUT_FILE"
        echo "   Size: $FILE_SIZE"
        echo "   Questions: $NUM_QUESTIONS"
    fi
    
    # Show datastore info
    if [ -d "$DATASTORE_DIR" ]; then
        DATASTORE_SIZE=$(du -sh "$DATASTORE_DIR" | cut -f1)
        echo "   Datastore: $DATASTORE_DIR ($DATASTORE_SIZE)"
    fi
else
    echo "❌ PIPELINE FAILED"
    echo ""
    echo "💡 Check:"
    echo "   1. Error log: logs/knn_lm_optimized_*.err"
    echo "   2. If OOM: Reduce BATCH_SIZE to 512 or 384"
    echo "   3. Monitor: nvidia-smi"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit $EXIT_CODE