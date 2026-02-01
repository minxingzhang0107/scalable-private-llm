#!/usr/bin/env python3
"""
compute_name_embedding.py - A standalone script to generate and save author embeddings.

This script isolates the most resource-intensive part of the workflow:
1. Loads unique author names from the mapping file.
2. Uses the MistralEmbeddingExtractor to generate a vector for each name.
3. Saves the embeddings array to a .npy file.
4. Saves the corresponding ordered list of names to a .json file.
"""

import json
import numpy as np
import argparse
import os

# Reuse the embedding extractor from the provided file
# NOTE: Ensure 'dp_author_name_perturbation.py' exists
try:
    from src.name_perturbation.dp_author_name_perturbation import MistralEmbeddingExtractor
except ImportError:
    print("❌ Could not import 'MistralEmbeddingExtractor'. Make sure 'dp_author_name_perturbation.py' exists in src/name_perturbation directory.")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate and save author name embeddings.")
    parser.add_argument('--model-path', type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help='Path to the embedding model.')
    parser.add_argument('--author-mapping-path', type=str, required=True, help='Path to the JSON file containing author names.')
    parser.add_argument('--output-embeddings-path', type=str, default='author_embeddings.npy', help='Path to save the final embeddings .npy file.')
    parser.add_argument('--output-names-path', type=str, default='author_names_ordered.json', help='Path to save the ordered list of names.')

    args = parser.parse_args()

    print("🚀 Starting Standalone Embedding Generation")
    print("=" * 50)
    print(f"CONFIGURATION:")
    print(f"  - Model Path: {args.model_path}")
    print(f"  - Author Mapping: {args.author_mapping_path}")
    print(f"  - Embeddings Output: {args.output_embeddings_path}")
    print(f"  - Names Output: {args.output_names_path}")
    print("=" * 50)

    embedding_extractor = None
    try:
        # Step 1: Initialize the extractor
        print("🔍 Initializing MistralEmbeddingExtractor...")
        embedding_extractor = MistralEmbeddingExtractor(args.model_path, verbose=True)
        
        # Step 2: Load unique author names
        print(f"📥 Loading author names from {args.author_mapping_path}...")
        with open(args.author_mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        author_mapping = data.get('author_mapping', {})
        unique_authors = sorted(list(set(name for name in author_mapping.values() if name and name.strip())))
        print(f"🧠 Found {len(unique_authors)} unique author names.")
        
        # Step 3: Generate embeddings
        author_embeddings = embedding_extractor.get_author_embeddings_batch(unique_authors)
        
        if author_embeddings.shape[0] != len(unique_authors):
            raise RuntimeError("Mismatch between number of names and number of embeddings generated.")

        # Step 4: Save the embeddings and the corresponding names
        print(f"\n💾 Saving embeddings to {args.output_embeddings_path}...")
        np.save(args.output_embeddings_path, author_embeddings)
        
        print(f"💾 Saving ordered names to {args.output_names_path}...")
        with open(args.output_names_path, 'w', encoding='utf-8') as f:
            json.dump(unique_authors, f, indent=2)

        print("\n🎉 Success! Embeddings and names have been saved.")
        print(f"   - Embeddings shape: {author_embeddings.shape}")
        print(f"   - Number of names: {len(unique_authors)}")

    except Exception as e:
        print(f"\n❌ An error occurred during embedding generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if embedding_extractor:
            print("\n🧹 Cleaning up resources...")
            embedding_extractor.cleanup()

if __name__ == "__main__":
    main()