#!/usr/bin/env python
"""
Test script to compute L2 norms of all 50 author name embeddings
Use this to decide a good clipping threshold C
FIXED: Computes norms BEFORE normalization
"""

import os
import json
import torch
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

set_seed(42)
warnings.filterwarnings("ignore")

def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """Load model and tokenizer - same as main script"""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # Same as reference
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device

def load_author_names(mapping_file="dataset/private/tofu/author_names_mapping.json"):
    """Load author names from mapping file - same as main script"""
    print(f"Loading author names from: {mapping_file}")
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    author_mapping = data.get('author_mapping', {})
    unique_authors = set(author_mapping.values())
    author_names = [name for name in unique_authors if name and name.strip()]
    author_names.sort()  # Same sorting as main script
    
    print(f"Loaded {len(author_names)} unique author names")
    return author_names

def get_embedding_norms(model, tokenizer, author_names, device):
    """Extract embeddings and compute their L2 norms - FIXED VERSION"""
    print("Extracting embeddings and computing norms...")
    
    norms = []
    embeddings_info = []
    
    with torch.no_grad():
        for i, name in enumerate(author_names):
            print(f"  Processing {i+1}/{len(author_names)}: {name}")
            
            # Format input EXACTLY same as main script
            formatted_input = f"<s>[INST] Author name: {name.strip()} [/INST]"
            
            try:
                # Tokenize - same as main script
                encoding = tokenizer(
                    formatted_input,
                    truncation=True,
                    max_length=64,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Forward pass - same as main script
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                
                # Extract embedding - same as main script
                seq_length = attention_mask.sum(dim=1) - 1
                if seq_length.item() >= 0 and seq_length.item() < last_hidden_state.shape[1]:
                    embedding = last_hidden_state[0, seq_length.item()]
                    
                    # FIXED: Compute L2 norm BEFORE normalization
                    result_unnormalized = embedding.cpu().float().numpy()
                    norm = np.linalg.norm(result_unnormalized)
                    
                    norms.append(norm)
                    embeddings_info.append({
                        'name': name,
                        'norm': norm,
                        'embedding_shape': result_unnormalized.shape
                    })
                    
                    print(f"    Norm: {norm:.2f}")
                    
                else:
                    print(f"    Warning: Sequence length issue for {name}")
                    norms.append(0.0)
                    embeddings_info.append({
                        'name': name,
                        'norm': 0.0,
                        'embedding_shape': (0,)
                    })
                    
            except Exception as e:
                print(f"    Error processing {name}: {e}")
                norms.append(0.0)
                embeddings_info.append({
                    'name': name,
                    'norm': 0.0,
                    'embedding_shape': (0,),
                    'error': str(e)
                })
    
    return np.array(norms), embeddings_info

def analyze_norms(norms, embeddings_info):
    """Analyze norm distribution and suggest clipping thresholds"""
    print("\n" + "="*60)
    print("EMBEDDING NORM ANALYSIS")
    print("="*60)
    
    # Remove zero norms for statistics
    valid_norms = norms[norms > 0]
    
    print(f"Total embeddings: {len(norms)}")
    print(f"Valid embeddings: {len(valid_norms)}")
    print(f"Zero/invalid norms: {len(norms) - len(valid_norms)}")
    
    if len(valid_norms) == 0:
        print("No valid embeddings found!")
        return
    
    print(f"\nNorm Statistics:")
    print(f"  Min:    {np.min(valid_norms):.2f}")
    print(f"  Max:    {np.max(valid_norms):.2f}")
    print(f"  Mean:   {np.mean(valid_norms):.2f}")
    print(f"  Median: {np.median(valid_norms):.2f}")
    print(f"  Std:    {np.std(valid_norms):.2f}")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(valid_norms, p)
        print(f"  {p}th percentile: {value:.2f}")
    
    # Suggest clipping thresholds
    print(f"\nSuggested Clipping Thresholds (C):")
    
    # Conservative: clips very few embeddings
    conservative_c = np.percentile(valid_norms, 95)
    print(f"  Conservative (clips ~5%):  C = {conservative_c:.1f}")
    
    # Moderate: clips some outliers
    moderate_c = np.percentile(valid_norms, 90)
    print(f"  Moderate (clips ~10%):     C = {moderate_c:.1f}")
    
    # Aggressive: clips more embeddings
    aggressive_c = np.percentile(valid_norms, 75)
    print(f"  Aggressive (clips ~25%):   C = {aggressive_c:.1f}")
    
    # Show which embeddings would be clipped at different thresholds
    print(f"\nEmbeddings that would be clipped:")
    thresholds = [conservative_c, moderate_c, aggressive_c]
    threshold_names = ["Conservative", "Moderate", "Aggressive"]
    
    for thresh, name in zip(thresholds, threshold_names):
        clipped_count = np.sum(valid_norms > thresh)
        print(f"  {name} (C={thresh:.1f}): {clipped_count} embeddings")
        
        if clipped_count > 0 and clipped_count <= 10:
            clipped_indices = np.where(norms > thresh)[0]
            clipped_names = [embeddings_info[i]['name'] for i in clipped_indices]
            print(f"    Authors: {clipped_names}")
    
    # Show top 5 largest norms
    print(f"\nTop 5 largest embedding norms:")
    sorted_indices = np.argsort(norms)[::-1]
    for i, idx in enumerate(sorted_indices[:5]):
        info = embeddings_info[idx]
        print(f"  {i+1}. {info['name']}: {info['norm']:.2f}")
    
    # Show bottom 5 smallest norms (excluding zeros)
    print(f"\nTop 5 smallest embedding norms:")
    valid_indices = [i for i, norm in enumerate(norms) if norm > 0]
    valid_norms_with_idx = [(norms[i], i) for i in valid_indices]
    valid_norms_with_idx.sort()
    
    for i, (norm, idx) in enumerate(valid_norms_with_idx[:5]):
        info = embeddings_info[idx]
        print(f"  {i+1}. {info['name']}: {info['norm']:.2f}")
    
    print(f"\nRecommendation:")
    print(f"  Start with C = {moderate_c:.0f} (moderate clipping)")
    print(f"  This will clip ~10% of embeddings (the largest ones)")
    print(f"  Adjust based on your privacy vs utility trade-off:")
    print(f"    - Smaller C = more privacy (more clipping)")  
    print(f"    - Larger C = less privacy (less clipping)")
    print(f"  Expected C range: {aggressive_c:.0f} - {conservative_c:.0f}")

def main():
    print("EMBEDDING NORM ANALYSIS FOR CLIPPING THRESHOLD")
    print("="*60)
    print("Purpose: Analyze L2 norms of raw embeddings to choose clipping threshold C")
    print("Process: Same embedding extraction as main clustering script")
    
    # Load model and data
    try:
        model, tokenizer, device = load_model_and_tokenizer()
        author_names = load_author_names()
        
        # Extract embeddings and compute norms
        norms, embeddings_info = get_embedding_norms(model, tokenizer, author_names, device)
        
        # Analyze and suggest thresholds
        analyze_norms(norms, embeddings_info)
        
        # Save results
        results = {
            'author_names': author_names,
            'norms': norms.tolist(),
            'embeddings_info': embeddings_info,
            'statistics': {
                'min': float(np.min(norms[norms > 0])) if np.sum(norms > 0) > 0 else 0.0,
                'max': float(np.max(norms)),
                'mean': float(np.mean(norms[norms > 0])) if np.sum(norms > 0) > 0 else 0.0,
                'median': float(np.median(norms[norms > 0])) if np.sum(norms > 0) > 0 else 0.0,
                'std': float(np.std(norms[norms > 0])) if np.sum(norms > 0) > 0 else 0.0,
                'percentiles': {
                    50: float(np.percentile(norms[norms > 0], 50)) if np.sum(norms > 0) > 0 else 0.0,
                    75: float(np.percentile(norms[norms > 0], 75)) if np.sum(norms > 0) > 0 else 0.0,
                    90: float(np.percentile(norms[norms > 0], 90)) if np.sum(norms > 0) > 0 else 0.0,
                    95: float(np.percentile(norms[norms > 0], 95)) if np.sum(norms > 0) > 0 else 0.0,
                    99: float(np.percentile(norms[norms > 0], 99)) if np.sum(norms > 0) > 0 else 0.0,
                }
            }
        }
        
        output_file = "embedding_norms_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        print(f"\nNext steps:")
        print(f"1. Use the recommended C value in your main clustering script")
        print(f"2. Run: ./author_clustering_bash.sh --clip-threshold <C_VALUE>")
        print(f"3. Adjust C based on privacy/utility requirements")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()