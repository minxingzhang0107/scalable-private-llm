#!/usr/bin/env python
"""
Author Name Clustering with Differential Privacy
EXACTLY matching reference file approach with clustering twist:
1. Load 50 author names from mapping file (same as reference)
2. Get embeddings using Mistral 7B (same as reference)  
3. Apply constrained k-means clustering with DP noise
4. Replace author names in question/answer text (same as reference)

FIXED: Proper clipping BEFORE normalization + bfloat16 conversion
"""

import os
import json
import math
import torch
import numpy as np
import argparse
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import warnings
import time
import re
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)

# Set seed for reproducibility
set_seed(42)
warnings.filterwarnings("ignore")

def get_optimal_device():
    """Get the optimal device for training - same as reference"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU (will be very slow)")
        return torch.device('cpu')

    best_device = 0
    max_free_memory = 0

    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        free_memory = total_memory - allocated_memory

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = i

    torch.cuda.set_device(best_device)
    device = torch.device(f'cuda:{best_device}')

    print(f"🎯 Selected device: {device} ({max_free_memory / 1e9:.1f}GB free)")
    return device

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer - exact same as reference"""
    print(f"🤖 Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # FIX: Use bfloat16 for numerical stability
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Model loaded")
    return model, tokenizer

def load_author_mapping(mapping_path):
    """Load author mapping from JSON file - same as reference"""
    print(f"📥 Loading author mapping from {mapping_path}")
    
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Author mapping file not found: {mapping_path}")
    
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        author_mapping = data.get('author_mapping', {})
        
        if not author_mapping:
            raise ValueError("No 'author_mapping' found in JSON file")
        
        # Validate mapping format
        for key, value in author_mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(f"Invalid mapping format: {key} -> {value}")
            if not value.strip():
                print(f"⚠️ Empty author name for index {key}")
        
        print(f"✅ Loaded mapping for {len(author_mapping)} authors")
        print(f"   Sample authors: {list(author_mapping.values())[:5]}")
        
        return author_mapping
        
    except Exception as e:
        print(f"❌ Error loading author mapping: {e}")
        raise

def build_author_names_list(author_mapping):
    """Build list of all 50 author names - same as reference"""
    print("🏗️ Building author names list...")
    
    # Extract unique author names (values from mapping), filter empty ones - same as reference
    unique_authors = set(author_mapping.values())
    author_names = [name for name in unique_authors if name and name.strip()]
    author_names.sort()  # For consistent ordering
    
    print(f"   Unique author names: {len(author_names)}")
    if len(unique_authors) != len(author_names):
        print(f"   ⚠️ Filtered out {len(unique_authors) - len(author_names)} empty author names")
    
    return author_names

def get_name_embeddings(model, tokenizer, author_names, device, max_length=64):
    """Extract RAW embeddings for author names - FIXED: no normalization here"""
    print(f"🔢 Extracting RAW embeddings for {len(author_names)} author names...")
    
    embeddings = []
    
    with torch.no_grad():
        for i, name in enumerate(author_names):
            print(f"  Processing {i+1}/{len(author_names)}: {name}")
            
            # Format author name as instruction - similar to reference approach
            formatted_input = f"<s>[INST] Author name: {name.strip()} [/INST]"
            
            # Tokenize - same as reference
            try:
                encoding = tokenizer(
                    formatted_input,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
            except Exception as e:
                print(f"❌ Tokenization error for author '{name}': {e}")
                # Use zero embedding as fallback
                embedding_dim = model.config.hidden_size
                embeddings.append(np.zeros(embedding_dim))
                continue
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward pass with error handling - same as reference
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            except Exception as e:
                print(f"❌ Forward pass error for author '{name}': {e}")
                embedding_dim = model.config.hidden_size
                embeddings.append(np.zeros(embedding_dim))
                continue
            
            # Extract embedding from last hidden state - same as reference approach
            last_hidden_state = outputs.hidden_states[-1]  # Shape: [1, seq_len, hidden_dim]
            
            # Use last token's embedding (same as reference approach)
            seq_length = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            if seq_length.item() >= 0 and seq_length.item() < last_hidden_state.shape[1]:
                embedding = last_hidden_state[0, seq_length.item()]
                
                # FIXED: Get RAW embedding (no normalization yet)
                result = embedding.cpu().float().numpy()  # Fixed bfloat16 issue
                
                # Validate embedding
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    print(f"⚠️ Invalid embedding for author '{name}', using zero embedding")
                    embedding_dim = model.config.hidden_size
                    embeddings.append(np.zeros(embedding_dim))
                else:
                    embeddings.append(result)
            else:
                print(f"⚠️ Sequence length issue for author '{name}'")
                embedding_dim = model.config.hidden_size
                embeddings.append(np.zeros(embedding_dim))
    
    embeddings = np.array(embeddings)
    print(f"✅ Extracted RAW embeddings with shape: {embeddings.shape}")
    return embeddings

def clip_and_normalize_embeddings(embeddings, clip_threshold):
    """FIXED: Clip BEFORE normalization, then normalize"""
    print(f"✂️ Clipping and normalizing embeddings with threshold C = {clip_threshold}")
    
    processed_embeddings = []
    original_norms = []
    clipped_norms = []
    final_norms = []
    
    for i, embedding in enumerate(embeddings):
        original_norm = np.linalg.norm(embedding)
        original_norms.append(original_norm)
        
        # Step 1: Clip raw embedding
        if original_norm > clip_threshold:
            clipped_embedding = embedding * (clip_threshold / original_norm)
            clipped_norm = clip_threshold
        else:
            clipped_embedding = embedding.copy()
            clipped_norm = original_norm
        
        clipped_norms.append(clipped_norm)
        
        # Step 2: Normalize to unit length
        clipped_embedding_norm = np.linalg.norm(clipped_embedding)
        if clipped_embedding_norm > 1e-12:
            normalized_embedding = clipped_embedding / clipped_embedding_norm
            final_norm = np.linalg.norm(normalized_embedding)
        else:
            # Handle edge case of zero embedding
            normalized_embedding = np.zeros_like(embedding)
            if len(normalized_embedding) > 0:
                normalized_embedding[0] = 1.0
            final_norm = 1.0
            
        processed_embeddings.append(normalized_embedding)
        final_norms.append(final_norm)
    
    processed_embeddings = np.array(processed_embeddings)
    
    print(f"  Original norms - Min: {np.min(original_norms):.2f}, Max: {np.max(original_norms):.2f}, Mean: {np.mean(original_norms):.2f}")
    print(f"  Clipped norms  - Min: {np.min(clipped_norms):.2f}, Max: {np.max(clipped_norms):.2f}, Mean: {np.mean(clipped_norms):.2f}")
    print(f"  Final norms    - Min: {np.min(final_norms):.4f}, Max: {np.max(final_norms):.4f}, Mean: {np.mean(final_norms):.4f}")
    print(f"  Clipped {np.sum(np.array(original_norms) > clip_threshold)} out of {len(embeddings)} embeddings")
    
    return processed_embeddings

def constrained_kmeans(embeddings, minimum_cluster_size):
    """
    Apply constraint K-means clustering: each cluster size ≥ minimum_cluster_size,
    maximize number of clusters to minimize accumulated distance to centroids
    """
    print(f"🎯 Running constrained k-means with minimum_cluster_size = {minimum_cluster_size}")
    
    n_samples = len(embeddings)
    max_possible_clusters = n_samples // minimum_cluster_size
    
    print(f"  Total samples: {n_samples}")
    print(f"  Theoretical maximum clusters: {max_possible_clusters}")
    
    if max_possible_clusters < 1:
        print("❌ Error: Not enough samples for even one cluster with the given minimum size")
        return None, None, 1
    
    # Initialize with k=1 (all points in one cluster - always valid)
    best_k = 1
    best_labels = np.zeros(n_samples, dtype=int)
    best_centroids = np.array([np.mean(embeddings, axis=0)])
    
    print(f"  Starting with k=1 (baseline): cluster size = {n_samples}")
    
    # Try increasing numbers of clusters to maximize k
    for k in range(2, max_possible_clusters + 1):
        print(f"  Trying k = {k} clusters...")
        
        # Run k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        
        # Check constraint: ALL clusters must have ≥ minimum_cluster_size samples
        cluster_sizes = np.bincount(labels)
        min_actual_cluster_size = np.min(cluster_sizes)
        
        print(f"    Cluster sizes: {cluster_sizes}")
        print(f"    Minimum actual cluster size: {min_actual_cluster_size}")
        
        if min_actual_cluster_size >= minimum_cluster_size:
            # Constraint satisfied - this k is valid, update best solution
            best_k = k
            best_labels = labels.copy()
            best_centroids = centroids.copy()
            
            # Calculate clustering quality metric (accumulated distance to centroids)
            total_wcss = sum([np.sum((embeddings[labels == i] - centroids[i])**2) 
                            for i in range(k)])
            
            print(f"    ✅ Constraint satisfied for k = {k}")
            print(f"    Within-cluster sum of squares: {total_wcss:.2f}")
        else:
            # Constraint violated - stop here (higher k will likely also violate)
            print(f"    ❌ Constraint violated for k = {k}")
            print(f"       Required: {minimum_cluster_size}, but smallest cluster has: {min_actual_cluster_size}")
            print(f"    Stopping search (higher k values will likely also violate constraint)")
            break
    
    # Final summary
    final_cluster_sizes = np.bincount(best_labels)
    final_wcss = sum([np.sum((embeddings[best_labels == i] - best_centroids[i])**2) 
                     for i in range(best_k)])
    
    print(f"✅ OPTIMAL SOLUTION FOUND:")
    print(f"  Number of clusters: {best_k} (maximized)")
    print(f"  Cluster sizes: {final_cluster_sizes}")
    print(f"  All clusters ≥ {minimum_cluster_size}: {np.all(final_cluster_sizes >= minimum_cluster_size)}")
    print(f"  Total within-cluster sum of squares: {final_wcss:.2f} (minimized)")
    print(f"  Average cluster size: {n_samples / best_k:.1f}")
    
    return best_labels, best_centroids, best_k

def add_dp_noise(centroids, clip_threshold, epsilon, delta):
    """Add differential privacy noise to centroids (depending on C, epsilon, delta)"""
    print(f"🔊 Adding DP noise with ε = {epsilon}, δ = {delta}, C = {clip_threshold}")
    
    # Calculate noise standard deviation for (ε, δ)-DP
    # For L2 sensitivity C: σ = C * sqrt(2 * ln(1.25/δ)) / ε
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("Delta must be in (0, 1)")
    
    sigma = clip_threshold * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    print(f"  Noise standard deviation: σ = {sigma:.6f}")
    
    noisy_centroids = []
    for i, centroid in enumerate(centroids):
        # Generate Gaussian noise
        noise = np.random.normal(0, sigma, size=centroid.shape)
        noisy_centroid = centroid + noise
        noisy_centroids.append(noisy_centroid)
        
        print(f"    Cluster {i}: Added noise with norm {np.linalg.norm(noise):.4f}")
    
    return np.array(noisy_centroids)

def find_nearest_names(noisy_centroids, original_embeddings, author_names):
    """Take 1NN from all 50 names to be the centroid name for each cluster"""
    print(f"🔍 Finding nearest names to {len(noisy_centroids)} noisy centroids...")
    print(f"  Searching among all {len(author_names)} author names")
    
    # Find nearest neighbors from all 50 names
    nearest_indices, distances = pairwise_distances_argmin_min(noisy_centroids, original_embeddings)
    
    centroid_names = []
    for i, (nearest_idx, distance) in enumerate(zip(nearest_indices, distances)):
        nearest_name = author_names[nearest_idx]
        centroid_names.append(nearest_name)
        print(f"  Cluster {i}: Nearest name = '{nearest_name}' (distance: {distance:.4f})")
    
    return centroid_names

def create_name_mapping(author_names, cluster_labels, centroid_names):
    """Create mapping from original names to centroid names"""
    print(f"📋 Creating name mapping...")
    
    name_mapping = {}
    cluster_assignments = defaultdict(list)
    
    # Group names by cluster and create mapping
    for name, label in zip(author_names, cluster_labels):
        cluster_assignments[label].append(name)
        name_mapping[name] = centroid_names[label]
    
    # Print cluster assignments
    print(f"📊 Cluster assignments:")
    for cluster_id in sorted(cluster_assignments.keys()):
        names_in_cluster = cluster_assignments[cluster_id]
        centroid_name = centroid_names[cluster_id]
        print(f"  Cluster {cluster_id} (→ '{centroid_name}'): {names_in_cluster}")
    
    return name_mapping

def get_author_for_qa_pair(qa_pair, author_mapping):
    """Get the original author name for a QA pair using author_index - same as reference"""
    author_index = str(qa_pair.get('author_index', 'unknown'))
    return author_mapping.get(author_index, "Unknown")

def replace_author_in_text(text, original_author, new_author):
    """Replace all occurrences of original author name with new author name in text - SAME AS REFERENCE"""
    if not text or not original_author or not new_author:
        return text
    
    original_author = original_author.strip()
    new_author = new_author.strip()
    
    if not original_author or not new_author:
        return text
    
    # Step 1: Replace exact full name matches (case-insensitive)
    # Use word boundaries to avoid partial matches
    pattern = r'\b' + re.escape(original_author) + r'\b'
    text = re.sub(pattern, new_author, text, flags=re.IGNORECASE)
    
    # Step 2: Handle individual name components intelligently
    original_parts = [part.strip() for part in original_author.split() if part.strip()]
    new_parts = [part.strip() for part in new_author.split() if part.strip()]
    
    # Only do component replacement if both names have multiple parts
    if len(original_parts) > 1 and len(new_parts) > 1:
        # Define words to skip (common articles, prepositions, etc.)
        skip_words = {
            'the', 'a', 'an', 'of', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'jr', 'sr', 'ii', 'iii', 'iv', 'v'  # suffixes
        }
        
        # Create mapping of original parts to new parts
        part_mapping = {}
        
        # Map first names (usually first part)
        if len(original_parts) >= 1 and len(new_parts) >= 1:
            orig_first = original_parts[0]
            new_first = new_parts[0]
            if len(orig_first) > 1 and orig_first.lower() not in skip_words:
                part_mapping[orig_first] = new_first
        
        # Map last names (usually last part)
        if len(original_parts) >= 2 and len(new_parts) >= 2:
            orig_last = original_parts[-1]
            new_last = new_parts[-1]
            if len(orig_last) > 1 and orig_last.lower() not in skip_words:
                part_mapping[orig_last] = new_last
        
        # Map middle parts if both names have them
        if len(original_parts) >= 3 and len(new_parts) >= 3:
            for i in range(1, min(len(original_parts)-1, len(new_parts)-1)):
                orig_middle = original_parts[i]
                new_middle = new_parts[i]
                if len(orig_middle) > 1 and orig_middle.lower() not in skip_words:
                    part_mapping[orig_middle] = new_middle
        
        # Apply the mappings
        for orig_part, new_part in part_mapping.items():
            if orig_part and new_part and len(orig_part) > 1:
                part_pattern = r'\b' + re.escape(orig_part) + r'\b'
                text = re.sub(part_pattern, new_part, text, flags=re.IGNORECASE)
    
    return text

def apply_name_mapping_to_training_data(train_file, author_mapping, name_mapping, output_file):
    """Apply name mapping to training data and save updated file - SAME AS REFERENCE"""
    print(f"🔄 Applying name mapping to training data...")
    print(f"  Input file: {train_file}")
    print(f"  Output file: {output_file}")
    
    # Load training data
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data file not found: {train_file}")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    if not isinstance(training_data, list):
        raise ValueError("Training data must be a list of examples")
    
    print(f"  Loaded {len(training_data)} training examples")
    
    # Apply name mapping - SAME PROCESS AS REFERENCE
    updated_data = []
    stats = {"same": 0, "changed": 0, "unknown": 0, "errors": 0}
    
    for qa_idx, qa_pair in enumerate(tqdm(training_data, desc="Processing QA pairs")):
        try:
            # Get original author - same as reference
            original_author = get_author_for_qa_pair(qa_pair, author_mapping)
            
            # Get mapped author name (noisy centroid name)
            if original_author in name_mapping:
                new_author = name_mapping[original_author]
            else:
                new_author = original_author  # Keep unchanged if not in mapping
            
            # Create new QA pair - same as reference
            new_qa_pair = qa_pair.copy()
            
            # Replace author names in question and answer - SAME AS REFERENCE
            try:
                new_qa_pair['question'] = replace_author_in_text(
                    qa_pair.get('question', ''), original_author, new_author
                )
                new_qa_pair['answer'] = replace_author_in_text(
                    qa_pair.get('answer', ''), original_author, new_author
                )
                if 'combined_text' in qa_pair:
                    new_qa_pair['combined_text'] = replace_author_in_text(
                        qa_pair.get('combined_text', ''), original_author, new_author
                    )
            except Exception as e:
                print(f"⚠️ Error replacing text in QA pair {qa_idx}: {e}")
                # Keep original text if replacement fails
                new_qa_pair['question'] = qa_pair.get('question', '')
                new_qa_pair['answer'] = qa_pair.get('answer', '')
                if 'combined_text' in qa_pair:
                    new_qa_pair['combined_text'] = qa_pair.get('combined_text', '')
            
            # Add metadata about perturbation - SAME AS REFERENCE
            new_qa_pair['original_author_name'] = original_author
            new_qa_pair['perturbed_author_name'] = new_author
            new_qa_pair['perturbation_applied'] = (original_author != new_author)
            new_qa_pair['qa_index'] = qa_idx
            
            updated_data.append(new_qa_pair)
            
            # Update statistics - same as reference
            if original_author == "Unknown":
                stats["unknown"] += 1
            elif original_author == new_author:
                stats["same"] += 1
            else:
                stats["changed"] += 1
            
            # Debug output for first few pairs
            if qa_idx < 5:
                print(f"   QA {qa_idx}: {original_author} → {new_author}")
                
        except Exception as e:
            print(f"⚠️ Error processing QA pair {qa_idx}: {e}")
            # Keep original QA pair
            error_qa_pair = qa_pair.copy()
            error_qa_pair['original_author_name'] = "Error"
            error_qa_pair['perturbed_author_name'] = "Error"
            error_qa_pair['perturbation_applied'] = False
            error_qa_pair['qa_index'] = qa_idx
            error_qa_pair['processing_error'] = str(e)
            updated_data.append(error_qa_pair)
            stats["errors"] += 1
    
    # Save updated training data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Updated training data saved!")
    print(f"  Total examples: {len(updated_data)}")
    print(f"  Mapping statistics:")
    print(f"    Unchanged: {stats['same']:,} QA pairs")
    print(f"    Changed: {stats['changed']:,} QA pairs")
    print(f"    Unknown: {stats['unknown']:,} QA pairs")
    print(f"    Errors: {stats['errors']:,} QA pairs")
    total_processed = stats['same'] + stats['changed']
    if total_processed > 0:
        print(f"    Change rate: {stats['changed']/total_processed*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Author Name Clustering with Differential Privacy")
    parser.add_argument('--model-name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Mistral model name')
    parser.add_argument('--author-mapping-file', type=str, required=True,
                        help='Path to author_names_mapping.json file')
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to training data file (tofu_train.json)')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output file for updated training data')
    parser.add_argument('--minimum-cluster-size', type=int, required=True,
                        help='Minimum number of authors per cluster')
    parser.add_argument('--clip-threshold', type=float, required=True,
                        help='L2 norm clipping threshold (C)')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='Privacy parameter epsilon')
    parser.add_argument('--delta', type=float, required=True,
                        help='Privacy parameter delta')
    
    args = parser.parse_args()
    
    print("🚀 AUTHOR NAME CLUSTERING WITH DIFFERENTIAL PRIVACY")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Author mapping file: {args.author_mapping_file}")
    print(f"Training data file: {args.train_file}")
    print(f"Output file: {args.output_file}")
    print(f"Minimum cluster size: {args.minimum_cluster_size}")
    print(f"Clipping threshold: {args.clip_threshold}")
    print(f"Privacy parameters: ε = {args.epsilon}, δ = {args.delta}")
    print("=" * 60)
    
    # Setup device (same as reference)
    device = get_optimal_device()
    torch.cuda.empty_cache()
    
    try:
        # Step 1: Load author mapping and extract all 50 names - SAME AS REFERENCE
        author_mapping = load_author_mapping(args.author_mapping_file)
        author_names = build_author_names_list(author_mapping)
        
        # Step 2: Load model - SAME AS REFERENCE
        model, tokenizer = load_model_and_tokenizer(args.model_name)
        
        # Move model to device (same as reference approach) 
        if not hasattr(model, 'quantization_config') or model.quantization_config is None:
            model = model.to(device)
        
        model.eval()  # Set to evaluation mode
        
        # Step 3: Take name embedding of all 50 author names using Mistral 7B
        raw_embeddings = get_name_embeddings(model, tokenizer, author_names, device)
        
        # Step 4: FIXED - Clip BEFORE normalization, then normalize
        processed_embeddings = clip_and_normalize_embeddings(raw_embeddings, args.clip_threshold)
        
        # Step 5: Apply constraint K-means clustering
        cluster_labels, centroids, num_clusters = constrained_kmeans(
            processed_embeddings, args.minimum_cluster_size
        )
        
        if cluster_labels is None:
            print("❌ Failed to find valid clustering")
            return 1
        
        # Step 6: For each cluster, find centroid vector, add DP noise
        noisy_centroids = add_dp_noise(centroids, args.clip_threshold, args.epsilon, args.delta)
        
        # Step 7: Take 1NN from all 50 names to be centroid name for each cluster
        centroid_names = find_nearest_names(noisy_centroids, processed_embeddings, author_names)
        
        # Step 8: Create mapping from original names to centroid names
        name_mapping = create_name_mapping(author_names, cluster_labels, centroid_names)
        
        # Step 9: Apply mapping to training data and save updated file - SAME AS REFERENCE
        apply_name_mapping_to_training_data(args.train_file, author_mapping, name_mapping, args.output_file)
        
        print(f"\n✅ SUCCESS: Author name clustering and data update completed!")
        
        # Summary
        unique_centroid_names = set(name_mapping.values())
        print(f"\n📊 FINAL SUMMARY:")
        print(f"  Original authors: {len(author_names)}")
        print(f"  Clusters created: {num_clusters}")
        print(f"  Unique centroid names: {len(unique_centroid_names)}")
        print(f"  Privacy budget used: ε = {args.epsilon}, δ = {args.delta}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup (same as reference)
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    exit(main())