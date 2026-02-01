#!/bin/bash

# =============================================================================
# TOFU Dataset Train/Test Split - No Paraphrasing
# - Randomly select 50 authors
# - Use all 20 QA pairs from selected authors as training (1000 total)
# - Randomly select 2 QA pairs from each author for testing (100 total)
# - Use original test data (no paraphrasing applied)
# =============================================================================

set -e

# Parameters
python_script="src/data_process/tofu_train_test_split.py"
log_file="logs/tofu_train_test_split.log"

# Create logs directory
mkdir -p logs

# Log basic info
echo "=== TOFU Modified Train/Test Split (Check Existing) Started ===" | tee "$log_file"
echo "Date: $(date)" | tee -a "$log_file"
echo "Host: $(hostname)" | tee -a "$log_file"

# Check Python script exists
if [ ! -f "$python_script" ]; then
    echo "❌ Error: $python_script not found!" | tee -a "$log_file"
    exit 1
fi

# Check GPU availability (not needed but for info)
if command -v nvidia-smi &> /dev/null; then
    echo "🔧 GPU Info (not needed for this script):" | tee -a "$log_file"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | tee -a "$log_file"
else
    echo "ℹ️  No GPU detected (not needed for this script)" | tee -a "$log_file"
fi

# Start processing
echo "🎯 Starting TOFU dataset check and split..." | tee -a "$log_file"
echo "Script: $python_script" | tee -a "$log_file"
echo "Behavior: Skip if train/test files already exist" | tee -a "$log_file"
echo "Method: Random 50 authors + original test data" | tee -a "$log_file"
echo "Output: dataset/private/tofu/" | tee -a "$log_file"
echo "Expected: 1000 train + 100 original test examples" | tee -a "$log_file"
echo "=========================================================" | tee -a "$log_file"

# Run processing with output capture
python -u "$python_script" 2>&1 | tee -a "$log_file"
exit_code=${PIPESTATUS[0]}

# Final status
echo "=========================================================" | tee -a "$log_file"
echo "Processing completed: $(date)" | tee -a "$log_file"

if [ $exit_code -eq 0 ]; then
    echo "✅ TOFU modified split (original data) successful!" | tee -a "$log_file"
    echo "📁 Data saved to: dataset/private/tofu/" | tee -a "$log_file"
    
    # Show detailed stats if available
    if [ -f "dataset/private/tofu/metadata.json" ]; then
        echo "📊 Detailed stats:" | tee -a "$log_file"
        python -c "
import json
try:
    with open('dataset/private/tofu/metadata.json', 'r') as f:
        metadata = json.load(f)
    print(f'  Selected authors: {metadata[\"total_authors_selected\"]}')
    print(f'  Train examples: {metadata[\"train_examples\"]}')
    print(f'  Test examples: {metadata[\"test_examples\"]} (100% original)')
    print(f'  No paraphrasing applied: {not metadata.get(\"paraphrasing_applied\", True)}')
    print(f'  Method: {metadata[\"split_method\"]}')
    
    # Show sample selected authors
    sample_authors = metadata['selected_authors'][:5]
    print(f'  Sample selected authors: {sample_authors}')
    
except Exception as e:
    print(f'  (Stats unavailable: {e})')
" | tee -a "$log_file"
    fi
    
    # Verify file sizes
    if [ -f "dataset/private/tofu/tofu_train.json" ] && [ -f "dataset/private/tofu/tofu_test.json" ]; then
        echo "📁 File verification:" | tee -a "$log_file"
        echo "  $(wc -l < dataset/private/tofu/tofu_train.json) lines in train file" | tee -a "$log_file"
        echo "  $(wc -l < dataset/private/tofu/tofu_test.json) lines in test file" | tee -a "$log_file"
        
        # Quick JSON validation
        python -c "
import json
try:
    with open('dataset/private/tofu/tofu_train.json') as f:
        train_data = json.load(f)
    with open('dataset/private/tofu/tofu_test.json') as f:
        test_data = json.load(f)
    
    print(f'  Train data: {len(train_data)} examples loaded successfully')
    print(f'  Test data: {len(test_data)} examples loaded successfully')
    
    # Check if test data has paraphrase info (should be False now)
    has_paraphrase_info = any('paraphrase_info' in item for item in test_data[:5])
    print(f'  Test data paraphrased: {\"❌\" if not has_paraphrase_info else \"✅\"} (should be ❌)')
    print(f'  Test data original: {\"✅\" if not has_paraphrase_info else \"❌\"} (should be ✅)')
    
except Exception as e:
    print(f'  JSON validation failed: {e}')
" | tee -a "$log_file"
    fi
    
else
    echo "❌ TOFU modified split (original data) failed (exit: $exit_code)" | tee -a "$log_file"
fi

exit $exit_code