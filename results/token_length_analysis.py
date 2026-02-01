import json
import numpy as np
from transformers import AutoTokenizer

def analyze_answer_lengths(json_file_path):
    """
    Analyze and compute average lengths for ground truth and generated answers
    
    Args:
        json_file_path: Path to the JSON file containing the data
    """
    
    # Load tokenizer for token analysis
    print("🤖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Analyzing {len(data)} Q&A pairs from: {json_file_path}")
    print("=" * 80)
    
    # Initialize lists to store lengths
    gt_char_lengths = []
    gen_char_lengths = []
    gt_word_lengths = []
    gen_word_lengths = []
    gt_token_lengths = []
    gen_token_lengths = []
    
    # Process each entry
    for i, entry in enumerate(data):
        gt_answer = entry.get('ground_truth_answer', '')
        gen_answer = entry.get('generated_answer', '')
        
        # Character lengths
        gt_char_len = len(gt_answer)
        gen_char_len = len(gen_answer)
        
        # Word lengths (split by whitespace)
        gt_word_len = len(gt_answer.split())
        gen_word_len = len(gen_answer.split())
        
        # Token lengths
        gt_token_len = len(tokenizer.encode(gt_answer))
        gen_token_len = len(tokenizer.encode(gen_answer))
        
        # Store lengths
        gt_char_lengths.append(gt_char_len)
        gen_char_lengths.append(gen_char_len)
        gt_word_lengths.append(gt_word_len)
        gen_word_lengths.append(gen_word_len)
        gt_token_lengths.append(gt_token_len)
        gen_token_lengths.append(gen_token_len)
        
        # Print first few examples for verification
        if i < 3:
            print(f"\nExample {i+1}:")
            print(f"  Ground Truth: {gt_answer[:100]}...")
            print(f"  Generated:    {gen_answer[:100]}...")
            print(f"  GT Length: {gt_char_len} chars, {gt_word_len} words, {gt_token_len} tokens")
            print(f"  Gen Length: {gen_char_len} chars, {gen_word_len} words, {gen_token_len} tokens")
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("📈 LENGTH STATISTICS")
    print("=" * 80)
    
    # Character-based statistics
    print("\n🔤 CHARACTER-BASED LENGTHS:")
    print(f"  Ground Truth:")
    print(f"    Average: {np.mean(gt_char_lengths):.2f} characters")
    print(f"    Median:  {np.median(gt_char_lengths):.2f} characters")
    print(f"    Min:     {np.min(gt_char_lengths)} characters")
    print(f"    Max:     {np.max(gt_char_lengths)} characters")
    print(f"    Std Dev: {np.std(gt_char_lengths):.2f} characters")
    
    print(f"\n  Generated Answer:")
    print(f"    Average: {np.mean(gen_char_lengths):.2f} characters")
    print(f"    Median:  {np.median(gen_char_lengths):.2f} characters")
    print(f"    Min:     {np.min(gen_char_lengths)} characters")
    print(f"    Max:     {np.max(gen_char_lengths)} characters")
    print(f"    Std Dev: {np.std(gen_char_lengths):.2f} characters")
    
    # Word-based statistics
    print("\n📝 WORD-BASED LENGTHS:")
    print(f"  Ground Truth:")
    print(f"    Average: {np.mean(gt_word_lengths):.2f} words")
    print(f"    Median:  {np.median(gt_word_lengths):.2f} words")
    print(f"    Min:     {np.min(gt_word_lengths)} words")
    print(f"    Max:     {np.max(gt_word_lengths)} words")
    print(f"    Std Dev: {np.std(gt_word_lengths):.2f} words")
    
    print(f"\n  Generated Answer:")
    print(f"    Average: {np.mean(gen_word_lengths):.2f} words")
    print(f"    Median:  {np.median(gen_word_lengths):.2f} words")
    print(f"    Min:     {np.min(gen_word_lengths)} words")
    print(f"    Max:     {np.max(gen_word_lengths)} words")
    print(f"    Std Dev: {np.std(gen_word_lengths):.2f} words")
    
    # Token-based statistics
    print("\n🔢 TOKEN-BASED LENGTHS:")
    print(f"  Ground Truth:")
    print(f"    Average: {np.mean(gt_token_lengths):.2f} tokens")
    print(f"    Median:  {np.median(gt_token_lengths):.2f} tokens")
    print(f"    Min:     {np.min(gt_token_lengths)} tokens")
    print(f"    Max:     {np.max(gt_token_lengths)} tokens")
    print(f"    Std Dev: {np.std(gt_token_lengths):.2f} tokens")
    
    print(f"\n  Generated Answer:")
    print(f"    Average: {np.mean(gen_token_lengths):.2f} tokens")
    print(f"    Median:  {np.median(gen_token_lengths):.2f} tokens")
    print(f"    Min:     {np.min(gen_token_lengths)} tokens")
    print(f"    Max:     {np.max(gen_token_lengths)} tokens")
    print(f"    Std Dev: {np.std(gen_token_lengths):.2f} tokens")
    
    # Comparison
    print("\n⚖️  COMPARISON:")
    char_ratio = np.mean(gen_char_lengths) / np.mean(gt_char_lengths)
    word_ratio = np.mean(gen_word_lengths) / np.mean(gt_word_lengths)
    token_ratio = np.mean(gen_token_lengths) / np.mean(gt_token_lengths)
    
    print(f"  Generated vs Ground Truth (Character ratio): {char_ratio:.2f}x")
    print(f"  Generated vs Ground Truth (Word ratio): {word_ratio:.2f}x")
    print(f"  Generated vs Ground Truth (Token ratio): {token_ratio:.2f}x")
    
    if token_ratio > 1.1:
        print("  📈 Generated answers are significantly longer")
    elif token_ratio < 0.9:
        print("  📉 Generated answers are significantly shorter")
    else:
        print("  ✅ Generated answers are similar in length")
    
    # Return summary statistics
    return {
        'ground_truth': {
            'avg_chars': np.mean(gt_char_lengths),
            'avg_words': np.mean(gt_word_lengths),
            'avg_tokens': np.mean(gt_token_lengths),
            'median_chars': np.median(gt_char_lengths),
            'median_words': np.median(gt_word_lengths),
            'median_tokens': np.median(gt_token_lengths),
            'std_chars': np.std(gt_char_lengths),
            'std_words': np.std(gt_word_lengths),
            'std_tokens': np.std(gt_token_lengths)
        },
        'generated': {
            'avg_chars': np.mean(gen_char_lengths),
            'avg_words': np.mean(gen_word_lengths),
            'avg_tokens': np.mean(gen_token_lengths),
            'median_chars': np.median(gen_char_lengths),
            'median_words': np.median(gen_word_lengths),
            'median_tokens': np.median(gen_token_lengths),
            'std_chars': np.std(gen_char_lengths),
            'std_words': np.std(gen_word_lengths),
            'std_tokens': np.std(gen_token_lengths)
        },
        'ratios': {
            'char_ratio': char_ratio,
            'word_ratio': word_ratio,
            'token_ratio': token_ratio
        }
    }

def simple_length_analysis(json_file_path):
    """
    Simple version that just prints the key averages including tokens
    """
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_lengths = [len(entry.get('ground_truth_answer', '')) for entry in data]
    gen_lengths = [len(entry.get('generated_answer', '')) for entry in data]
    
    gt_word_lengths = [len(entry.get('ground_truth_answer', '').split()) for entry in data]
    gen_word_lengths = [len(entry.get('generated_answer', '').split()) for entry in data]
    
    gt_token_lengths = [len(tokenizer.encode(entry.get('ground_truth_answer', ''))) for entry in data]
    gen_token_lengths = [len(tokenizer.encode(entry.get('generated_answer', ''))) for entry in data]
    
    print(f"📊 Analysis of {len(data)} Q&A pairs:")
    print(f"  Ground Truth - Avg: {np.mean(gt_lengths):.1f} chars, {np.mean(gt_word_lengths):.1f} words, {np.mean(gt_token_lengths):.1f} tokens")
    print(f"  Generated    - Avg: {np.mean(gen_lengths):.1f} chars, {np.mean(gen_word_lengths):.1f} words, {np.mean(gen_token_lengths):.1f} tokens")
    print(f"  Ratio: {np.mean(gen_lengths)/np.mean(gt_lengths):.2f}x (chars), {np.mean(gen_word_lengths)/np.mean(gt_word_lengths):.2f}x (words), {np.mean(gen_token_lengths)/np.mean(gt_token_lengths):.2f}x (tokens)")

# Direct usage with hardcoded file path
if __name__ == "__main__":
    # Hardcoded file path
    json_file_path = "private/tofu/1nn_lm_pretrained_embedding/evaluation_combined_lm_1nn_generated_answers_pretrained_embedding_threshold_0.4_60_30.json"
    
    print("🚀 Running analysis with TOKEN COUNTS...")
    
    # Run detailed analysis including tokens
    stats = analyze_answer_lengths(json_file_path)
    
    print("\n✅ Analysis complete with token counts!")