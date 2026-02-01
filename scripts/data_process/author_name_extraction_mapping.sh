#!/bin/bash

# =============================================================================
# TOFU Author Name Extraction and Training Data Verification
# =============================================================================

set -e

# Parameters
python_script="src/data_process/author_name_extraction_mapping.py"
log_file="logs/author_extract.log"

# Create logs directory
mkdir -p logs

# Log basic info
echo "=== TOFU Author Extraction Started ===" | tee "$log_file"
echo "Date: $(date)" | tee -a "$log_file"
echo "Host: $(hostname)" | tee -a "$log_file"

# Check Python script exists
if [ ! -f "$python_script" ]; then
    echo "❌ Error: $python_script not found!" | tee -a "$log_file"
    exit 1
fi

# Check if processed training data exists
if [ ! -f "dataset/private/tofu/tofu_train.json" ]; then
    echo "❌ Error: dataset/private/tofu/tofu_train.json not found!" | tee -a "$log_file"
    echo "Please run bash script/data_process/tofu_train_test_split.sh first to generate the training data." | tee -a "$log_file"
    exit 1
fi

# Start processing
echo "🎯 Starting author name extraction and verification..." | tee -a "$log_file"
echo "Script: $python_script" | tee -a "$log_file"
echo "Input: dataset/private/tofu/tofu_train.json" | tee -a "$log_file"
echo "Output: dataset/private/tofu/author_names_mapping.json" | tee -a "$log_file"
echo "        dataset/private/tofu/author_analysis.json" | tee -a "$log_file"
echo "=========================================================" | tee -a "$log_file"

# Run processing with output capture
python -u "$python_script" 2>&1 | tee -a "$log_file"
exit_code=${PIPESTATUS[0]}

# Final status
echo "=========================================================" | tee -a "$log_file"
echo "Processing completed: $(date)" | tee -a "$log_file"

if [ $exit_code -eq 0 ]; then
    echo "✅ Author extraction successful!" | tee -a "$log_file"
    echo "📁 Results saved to:" | tee -a "$log_file"
    echo "    - dataset/private/tofu/author_names_mapping.json" | tee -a "$log_file"
    echo "    - dataset/private/tofu/author_analysis.json" | tee -a "$log_file"
    
    # Show quick stats if available
    if [ -f "dataset/private/tofu/author_names_mapping.json" ]; then
        echo "📊 Quick stats:" | tee -a "$log_file"
        python -c "
import json
try:
    with open('dataset/private/tofu/author_names_mapping.json', 'r') as f:
        mapping = json.load(f)
    print(f'  Total authors extracted: {mapping[\"total_authors\"]}')
    print(f'  Examples per author: {mapping[\"examples_per_author\"]}')
    
    # Show first few author names
    authors = list(mapping['author_mapping'].items())[:5]
    print(f'  Sample authors:')
    for author_id, name in authors:
        print(f'    {author_id} = {name}')
    if len(mapping['author_mapping']) > 5:
        print(f'    ... and {len(mapping[\"author_mapping\"]) - 5} more')
        
except Exception as e:
    print(f'  (Stats unavailable: {e})')
" | tee -a "$log_file"
    fi
    
    if [ -f "dataset/private/tofu/author_analysis.json" ]; then
        echo "📋 Training data analysis completed!" | tee -a "$log_file"
        python -c "
import json
try:
    with open('dataset/private/tofu/author_analysis.json', 'r') as f:
        analysis = json.load(f)
    analyzed = analysis['summary_statistics']['authors_analyzed']
    print(f'  Authors analyzed in detail: {analyzed}')
    print(f'  QA pairs per author in training: {analysis[\"summary_statistics\"][\"examples_per_author\"]}')
except Exception as e:
    print(f'  (Analysis stats unavailable: {e})')
" | tee -a "$log_file"
    fi
else
    echo "❌ Author extraction failed (exit: $exit_code)" | tee -a "$log_file"
fi

echo "🏁 Script completed" | tee -a "$log_file"
exit $exit_code