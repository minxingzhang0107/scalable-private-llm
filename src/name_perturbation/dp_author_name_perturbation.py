#!/usr/bin/env python3
"""
dp_author_name_perturbation.py - Fixed Differential Privacy Author Name Perturbation
Uses Mistral-7B for embeddings and processes JSON format with author_index

FIXES APPLIED:
- Thread-safe embedding extraction
- Robust hook setup with systematic fallback
- Better memory management
- Improved error handling
- Fixed text replacement logic
- Better random state management

NO LOGICAL CHANGES TO DP MECHANISM OR PERTURBATION LOGIC!
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
import threading
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import gc
warnings.filterwarnings("ignore")

class MistralEmbeddingExtractor:
    """Extract embeddings for author names using Mistral-7B model"""
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = None, verbose: bool = True):
        # Force GPU usage if available
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                if verbose:
                    print(f"🚀 CUDA available! Using GPU: {torch.cuda.get_device_name()}")
                    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                self.device = torch.device('cpu')
                print("⚠️  CUDA not available, using CPU (this will be slow!)")
        else:
            self.device = torch.device(device)
        
        self.verbose = verbose
        
        if self.verbose:
            print(f"🔍 Loading Mistral-7B model for author embeddings: {model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with memory optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=False
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
                
            if self.verbose:
                print("✅ Successfully loaded Mistral-7B model")
                        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Enable GPU optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            
            if self.verbose:
                print(f"✅ Model loaded on GPU")
                print(f"   GPU optimizations enabled")
        else:
            if self.verbose:
                print(f"✅ Model loaded on CPU")
                print("   ⚠️  This will be significantly slower than GPU!")
        
        # FIXED: Thread-local storage for embeddings
        self.thread_local = threading.local()
        self.hook_handles = []
        
        # Setup embedding hook with better error handling
        self._setup_embedding_hook()
        
        if self.verbose:
            print(f"✅ Model loaded on {self.device}")
            print(f"   Embedding dimension: {self.model.config.hidden_size}")
    
    def _setup_embedding_hook(self):
        """FIXED: Robust hook setup with systematic fallback"""
        if self.verbose:
            print("🔍 Setting up embedding hook...")
        
        # Clear existing hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        # Systematic search for hook location with comprehensive fallbacks
        hook_candidates = [
            ("model.layers[-1].post_attention_layernorm", 
             lambda: self.model.model.layers[-1].post_attention_layernorm),
            ("layers[-1].post_attention_layernorm", 
             lambda: self.model.layers[-1].post_attention_layernorm),
            ("model.layers[-1].input_layernorm", 
             lambda: self.model.model.layers[-1].input_layernorm),
            ("layers[-1].input_layernorm", 
             lambda: self.model.layers[-1].input_layernorm),
            ("model.layers[-1]", 
             lambda: self.model.model.layers[-1]),
            ("layers[-1]", 
             lambda: self.model.layers[-1]),
        ]
        
        target_module = None
        hook_location = None
        
        for location_name, get_module_fn in hook_candidates:
            try:
                candidate_module = get_module_fn()
                if candidate_module is not None:
                    target_module = candidate_module
                    hook_location = location_name
                    if self.verbose:
                        print(f"✅ Found hook location: {location_name}")
                    break
            except (AttributeError, IndexError) as e:
                if self.verbose:
                    print(f"   Tried {location_name}: failed")
                continue
        
        if target_module is None:
            raise RuntimeError("Could not find any suitable hook location in model")
        
        def hook_fn(module, input, output):
            try:
                if output is not None:
                    # FIXED: Use thread-local storage
                    if not hasattr(self.thread_local, 'embeddings'):
                        self.thread_local.embeddings = []
                    self.thread_local.embeddings.append(output.detach().cpu())
            except Exception as e:
                print(f"Hook error: {e}")
        
        handle = target_module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)
        
        if self.verbose:
            print("✅ Hook registered successfully")
    
    def _clear_gpu_cache(self):
        """Aggressive GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def get_author_embedding(self, author_name: str, max_length: int = 64) -> np.ndarray:
        """FIXED: Get embedding for a single author name with improved error handling"""
        if not author_name or not author_name.strip():
            # Return zero embedding for empty names
            embedding_dim = self.model.config.hidden_size
            return np.zeros(embedding_dim)
        
        with torch.no_grad():
            try:
                # FIXED: Clear thread-local embeddings before processing
                self.thread_local.embeddings = []
                
                # Format author name as instruction
                formatted_input = f"<s>[INST] Author name: {author_name.strip()} [/INST]"
                
                # Tokenize with error handling
                try:
                    encoding = self.tokenizer(
                        formatted_input,
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt'
                    )
                except Exception as e:
                    print(f"❌ Tokenization error for author '{author_name}': {e}")
                    embedding_dim = self.model.config.hidden_size
                    return np.zeros(embedding_dim)
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Forward pass with error handling
                try:
                    with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"❌ Forward pass error for author '{author_name}': {e}")
                    embedding_dim = self.model.config.hidden_size
                    return np.zeros(embedding_dim)
                
                # FIXED: Extract embedding from thread-local hook storage
                if hasattr(self.thread_local, 'embeddings') and self.thread_local.embeddings:
                    batch_embeddings = self.thread_local.embeddings[0]  # Shape: [1, seq_len, hidden_dim]
                    
                    # Use last token's embedding
                    seq_length = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                    if seq_length.item() < batch_embeddings.shape[1]:
                        embedding = batch_embeddings[0, seq_length.item()]
                        
                        # L2 normalize the embedding
                        embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
                        result = embedding.squeeze().numpy()
                        
                        # Clear GPU memory after each extraction
                        self._clear_gpu_cache()
                        
                        return result
                    else:
                        print(f"⚠️  Sequence length issue for author '{author_name}'")
                        embedding_dim = self.model.config.hidden_size
                        return np.zeros(embedding_dim)
                else:
                    print(f"❌ No embeddings extracted for author '{author_name}'")
                    embedding_dim = self.model.config.hidden_size
                    return np.zeros(embedding_dim)
            
            except Exception as e:
                print(f"❌ Error extracting embedding for author: {author_name}: {e}")
                embedding_dim = self.model.config.hidden_size
                return np.zeros(embedding_dim)
    
    def get_author_embeddings_batch(self, author_names: List[str], max_length: int = 64, batch_size: int = 8) -> np.ndarray:
        """FIXED: Get embeddings for multiple author names with better memory management"""
        all_embeddings = []
        
        if self.verbose:
            print(f"🧠 Extracting embeddings for {len(author_names)} author names (batch_size={batch_size})...")
        
        # FIXED: Process in smaller batches to manage memory better
        for batch_start in range(0, len(author_names), batch_size):
            batch_end = min(batch_start + batch_size, len(author_names))
            batch_names = author_names[batch_start:batch_end]
            
            batch_embeddings = []
            for author_name in batch_names:
                try:
                    embedding = self.get_author_embedding(author_name, max_length)
                    
                    # Validate embedding
                    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                        print(f"⚠️  Invalid embedding for author '{author_name}', using zero embedding")
                        embedding_dim = self.model.config.hidden_size
                        embedding = np.zeros(embedding_dim)
                    
                    batch_embeddings.append(embedding)
                        
                except Exception as e:
                    print(f"⚠️  Error with author '{author_name}': {e}")
                    # Use zero embedding as fallback
                    embedding_dim = self.model.config.hidden_size
                    zero_embedding = np.zeros(embedding_dim)
                    batch_embeddings.append(zero_embedding)
            
            all_embeddings.extend(batch_embeddings)
            
            # Clear GPU cache after each batch
            self._clear_gpu_cache()
            
            if self.verbose and len(author_names) > batch_size:
                batch_num = batch_start // batch_size + 1
                total_batches = (len(author_names) - 1) // batch_size + 1
                print(f"   Processed batch {batch_num}/{total_batches}")
        
        # Handle edge case: if no valid embeddings
        if not all_embeddings:
            print("❌ No valid embeddings extracted!")
            return np.array([]).reshape(0, self.model.config.hidden_size)
        
        result = np.vstack(all_embeddings)
        if self.verbose:
            print(f"✅ Extracted {result.shape[0]} author embeddings of dimension {result.shape[1]}")
        return result
    
    def cleanup(self):
        """Remove hooks and free memory"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            
        # Aggressive GPU memory cleanup
        self._clear_gpu_cache()
        
        if self.verbose:
            print("🧹 Mistral embedding extractor cleaned up")
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                print(f"   GPU memory still allocated: {memory_allocated:.2f} GB")

class DifferentialPrivacyPerturbator:
    """Add differential privacy noise to embeddings using standard Gaussian mechanism"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, verbose: bool = True):
        """
        Args:
            epsilon: Privacy parameter (smaller = more privacy, more noise)
            delta: Failure probability for (ε, δ)-DP
            verbose: Print verbose output
        """
        self.epsilon = epsilon
        self.delta = delta
        self.verbose = verbose
        
        # Standard sensitivity for L2-normalized embeddings
        # For unit vectors, the L2 sensitivity is at most 2 (when vectors are opposite)
        self.sensitivity = 2.0
        
        # Standard Gaussian mechanism noise scale
        # σ = sensitivity * sqrt(2 * log(1.25/δ)) / ε
        self.noise_scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # FIXED: Create separate random generator for DP operations
        self.rng = np.random.RandomState()
        
        if self.verbose:
            print(f"🔒 Standard Differential Privacy Perturbator initialized")
            print(f"   Privacy parameter ε: {epsilon}")
            print(f"   Failure probability δ: {delta}")
            print(f"   L2 sensitivity: {self.sensitivity}")
            print(f"   Gaussian noise scale σ: {self.noise_scale:.4f}")
            print(f"   Privacy guarantee: ({epsilon}, {delta})-DP")
    
    def add_noise(self, embedding: np.ndarray, random_seed: Optional[int] = None) -> np.ndarray:
        """FIXED: Add standard Gaussian DP noise to an embedding vector"""
        # FIXED: Use isolated random state for reproducibility
        if random_seed is not None:
            temp_rng = np.random.RandomState(random_seed)
        else:
            temp_rng = self.rng
        
        # Validate input
        if embedding.shape[0] == 0:
            return embedding.copy()
        
        # Generate i.i.d. Gaussian noise for each dimension
        noise = temp_rng.normal(
            loc=0.0,
            scale=self.noise_scale,
            size=embedding.shape
        )
        
        # Add noise to embedding
        noisy_embedding = embedding + noise
        
        # FIXED: Re-normalize to unit sphere with robust handling of edge cases
        norm = np.linalg.norm(noisy_embedding)
        if norm > 1e-12:  # Use smaller threshold for better precision
            noisy_embedding = noisy_embedding / norm
        else:
            # If norm is effectively zero, generate a random unit vector
            random_vector = temp_rng.normal(0, 1, embedding.shape)
            random_norm = np.linalg.norm(random_vector)
            if random_norm > 1e-12:
                noisy_embedding = random_vector / random_norm
            else:
                # Ultimate fallback: create unit vector in first dimension
                noisy_embedding = np.zeros_like(embedding)
                if len(noisy_embedding) > 0:
                    noisy_embedding[0] = 1.0
        
        return noisy_embedding

class AuthorDatasetPerturbator:
    """Main class for perturbing author names in the TOFU dataset using JSON format"""
    
    def __init__(self, 
                 model_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 verbose: bool = True):
        self.verbose = verbose
        
        # Initialize components
        self.embedding_extractor = MistralEmbeddingExtractor(model_path, verbose=verbose)
        self.dp_perturbator = DifferentialPrivacyPerturbator(epsilon, delta, verbose=verbose)
        
        # Storage for author embeddings
        self.author_names = []
        self.author_embeddings = None
        self.author_to_idx = {}
    
    def load_author_mapping(self, mapping_path: str) -> Dict[str, str]:
        """Load author mapping from JSON file with validation"""
        if self.verbose:
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
                    print(f"⚠️  Empty author name for index {key}")
            
            if self.verbose:
                print(f"✅ Loaded mapping for {len(author_mapping)} authors")
                print(f"   Sample authors: {list(author_mapping.values())[:5]}")
            
            return author_mapping
            
        except Exception as e:
            print(f"❌ Error loading author mapping: {e}")
            raise
    
    def build_author_embedding_space(self, author_mapping: Dict[str, str]):
        """Build embedding space for all author names"""
        if self.verbose:
            print("🏗️  Building author embedding space...")
        
        # Extract unique author names (values from mapping), filter empty ones
        unique_authors = set(author_mapping.values())
        self.author_names = [name for name in unique_authors if name and name.strip()]
        self.author_names.sort()  # For consistent ordering
        
        if self.verbose:
            print(f"   Unique author names: {len(self.author_names)}")
            if len(unique_authors) != len(self.author_names):
                print(f"   ⚠️  Filtered out {len(unique_authors) - len(self.author_names)} empty author names")
        
        # Create author to index mapping
        self.author_to_idx = {name: idx for idx, name in enumerate(self.author_names)}
        
        # Extract embeddings for all author names
        self.author_embeddings = self.embedding_extractor.get_author_embeddings_batch(self.author_names)
        
        if self.verbose:
            print(f"✅ Author embedding space built: {self.author_embeddings.shape}")
    
    def find_nearest_author(self, noisy_embedding: np.ndarray) -> str:
        """FIXED: Find nearest author name to a noisy embedding with comprehensive error handling"""
        # Input validation
        if self.author_embeddings is None or self.author_embeddings.shape[0] == 0:
            return "Unknown"
        
        if not self.author_names:
            return "Unknown"
        
        try:
            # Validate embedding dimensions
            expected_dim = self.author_embeddings.shape[1]
            if noisy_embedding.shape[0] != expected_dim:
                print(f"⚠️  Dimension mismatch: got {noisy_embedding.shape[0]}, expected {expected_dim}")
                return self.author_names[0]
            
            # Ensure input embedding is normalized and valid
            if np.any(np.isnan(noisy_embedding)) or np.any(np.isinf(noisy_embedding)):
                print("⚠️  Input embedding contains NaN/inf values")
                return self.author_names[0]
            
            # Normalize the noisy embedding
            norm = np.linalg.norm(noisy_embedding)
            if norm < 1e-12:
                print("⚠️  Input embedding has zero norm")
                return self.author_names[0]
            
            normalized_embedding = noisy_embedding / norm
            
            # Compute similarities with robust error handling
            try:
                similarities = cosine_similarity(
                    normalized_embedding.reshape(1, -1), 
                    self.author_embeddings
                )[0]
            except Exception as e:
                print(f"⚠️  Similarity computation failed: {e}")
                return self.author_names[0]
            
            # Validate similarities
            if similarities.shape[0] != len(self.author_names):
                print(f"⚠️  Similarity shape mismatch: {similarities.shape[0]} vs {len(self.author_names)}")
                return self.author_names[0]
            
            # Handle edge cases in similarities
            valid_mask = ~(np.isnan(similarities) | np.isinf(similarities))
            if not np.any(valid_mask):
                print("⚠️  All similarities are invalid")
                return self.author_names[0]
            
            # Find the best match among valid similarities
            valid_indices = np.where(valid_mask)[0]
            valid_similarities = similarities[valid_mask]
            
            best_valid_idx = np.argmax(valid_similarities)
            best_original_idx = valid_indices[best_valid_idx]
            
            # Final bounds check
            if 0 <= best_original_idx < len(self.author_names):
                return self.author_names[best_original_idx]
            else:
                print(f"⚠️  Invalid index: {best_original_idx}")
                return self.author_names[0]
                
        except Exception as e:
            print(f"⚠️  Unexpected error in nearest neighbor search: {e}")
            return self.author_names[0] if self.author_names else "Unknown"
    
    def get_author_for_qa_pair(self, qa_pair: dict, author_mapping: Dict[str, str]) -> str:
        """Get the original author name for a QA pair using author_index"""
        author_index = str(qa_pair.get('author_index', 'unknown'))
        return author_mapping.get(author_index, "Unknown")
    
    def perturb_author_for_qa_pair(self, qa_pair: dict, author_mapping: Dict[str, str], qa_idx: int) -> str:
        """Get perturbed author name for a given QA pair (independent for each QA pair)"""
        # Get original author
        original_author = self.get_author_for_qa_pair(qa_pair, author_mapping)
        
        if original_author == "Unknown" or original_author not in self.author_to_idx:
            return original_author  # Return as-is if not found
        
        # Get original embedding
        author_idx = self.author_to_idx[original_author]
        original_embedding = self.author_embeddings[author_idx]
        
        # Add DP noise (use qa_idx as seed for reproducibility)
        # Each QA pair gets independent noise!
        noisy_embedding = self.dp_perturbator.add_noise(original_embedding, random_seed=qa_idx)
        
        # Find nearest neighbor
        perturbed_author = self.find_nearest_author(noisy_embedding)
        
        return perturbed_author
    
    def replace_author_in_text(self, text: str, original_author: str, new_author: str) -> str:
        """FIXED: Replace all occurrences of original author name with new author name in text"""
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
    
    def process_dataset(self, 
                       train_json_path: str, 
                       author_mapping: Dict[str, str],
                       output_path: str):
        """Process the JSON dataset and create perturbed version"""
        if self.verbose:
            print(f"📖 Processing JSON dataset: {train_json_path}")
            print(f"💾 Output path: {output_path}")
        
        if not os.path.exists(train_json_path):
            raise FileNotFoundError(f"Training data file not found: {train_json_path}")
        
        # Read JSON data with error handling
        try:
            with open(train_json_path, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {e}")
        
        if not isinstance(qa_pairs, list):
            raise ValueError("JSON file should contain a list of QA pairs")
        
        if self.verbose:
            print(f"📊 Total QA pairs: {len(qa_pairs)}")
            unique_author_indices = set(str(qa.get('author_index', 'unknown')) for qa in qa_pairs)
            print(f"   Author indices in dataset: {len(unique_author_indices)}")
        
        # Validate QA pairs structure
        required_fields = ['question', 'answer', 'author_index']
        for i, qa_pair in enumerate(qa_pairs[:5]):  # Check first 5 pairs
            missing_fields = [field for field in required_fields if field not in qa_pair]
            if missing_fields:
                print(f"⚠️  QA pair {i} missing fields: {missing_fields}")
        
        # Process each QA pair INDEPENDENTLY
        perturbed_qa_pairs = []
        stats = {"same": 0, "changed": 0, "unknown": 0, "errors": 0}
        
        for qa_idx, qa_pair in enumerate(tqdm(qa_pairs, desc="Perturbing dataset", disable=not self.verbose)):
            try:
                # Get original and perturbed author names
                original_author = self.get_author_for_qa_pair(qa_pair, author_mapping)
                
                # INDEPENDENT perturbation for each QA pair!
                perturbed_author = self.perturb_author_for_qa_pair(qa_pair, author_mapping, qa_idx)
                
                # Create new QA pair with perturbed author names
                new_qa_pair = qa_pair.copy()
                
                # Replace author names in question and answer with error handling
                try:
                    new_qa_pair['question'] = self.replace_author_in_text(
                        qa_pair.get('question', ''), original_author, perturbed_author
                    )
                    new_qa_pair['answer'] = self.replace_author_in_text(
                        qa_pair.get('answer', ''), original_author, perturbed_author
                    )
                    if 'combined_text' in qa_pair:
                        new_qa_pair['combined_text'] = self.replace_author_in_text(
                            qa_pair.get('combined_text', ''), original_author, perturbed_author
                        )
                except Exception as e:
                    print(f"⚠️  Error replacing text in QA pair {qa_idx}: {e}")
                    # Keep original text if replacement fails
                    new_qa_pair['question'] = qa_pair.get('question', '')
                    new_qa_pair['answer'] = qa_pair.get('answer', '')
                    if 'combined_text' in qa_pair:
                        new_qa_pair['combined_text'] = qa_pair.get('combined_text', '')
                
                # Add metadata about perturbation
                new_qa_pair['original_author_name'] = original_author
                new_qa_pair['perturbed_author_name'] = perturbed_author
                new_qa_pair['perturbation_applied'] = (original_author != perturbed_author)
                new_qa_pair['qa_index'] = qa_idx
                
                perturbed_qa_pairs.append(new_qa_pair)
                
                # Update statistics
                if original_author == "Unknown":
                    stats["unknown"] += 1
                elif original_author == perturbed_author:
                    stats["same"] += 1
                else:
                    stats["changed"] += 1
                
                # Debug output for first few pairs
                if self.verbose and qa_idx < 5:
                    print(f"   QA {qa_idx}: {original_author} → {perturbed_author}")
                    
            except Exception as e:
                print(f"⚠️  Error processing QA pair {qa_idx}: {e}")
                # Keep original QA pair
                error_qa_pair = qa_pair.copy()
                error_qa_pair['original_author_name'] = "Error"
                error_qa_pair['perturbed_author_name'] = "Error"
                error_qa_pair['perturbation_applied'] = False
                error_qa_pair['qa_index'] = qa_idx
                error_qa_pair['processing_error'] = str(e)
                perturbed_qa_pairs.append(error_qa_pair)
                stats["errors"] += 1
        
        # Save perturbed dataset with error handling
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(perturbed_qa_pairs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise ValueError(f"Error saving output file: {e}")
        
        if self.verbose:
            print(f"✅ Dataset processing complete!")
            print(f"📊 Statistics:")
            print(f"   Unchanged: {stats['same']:,} QA pairs")
            print(f"   Changed: {stats['changed']:,} QA pairs")
            print(f"   Unknown: {stats['unknown']:,} QA pairs")
            print(f"   Errors: {stats['errors']:,} QA pairs")
            total_processed = stats['same'] + stats['changed']
            if total_processed > 0:
                print(f"   Change rate: {stats['changed']/total_processed*100:.1f}%")
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'embedding_extractor'):
            self.embedding_extractor.cleanup()

def main():
    """Main function with improved error handling"""
    parser = argparse.ArgumentParser(description='Add DP noise to author embeddings using Mistral-7B')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Path to Mistral model for embedding extraction')
    
    # Privacy parameters
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Privacy parameter (smaller = more privacy)')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='Failure probability for (ε, δ)-DP')
    
    # Data paths
    parser.add_argument('--train-json-path', type=str, 
                        default='dataset/private/tofu/tofu_train.json',
                        help='Path to training JSON file')
    parser.add_argument('--author-mapping-path', type=str,
                        default='dataset/private/tofu/author_names_mapping.json',
                        help='Path to author mapping JSON file')
    parser.add_argument('--output-path', type=str,
                        default='dataset/private/tofu/tofu_train_perturbed.json',
                        help='Path for output perturbed dataset')
    
    args = parser.parse_args()
    
    # GPU availability check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("🔒 FIXED Differential Privacy Author Name Perturbation (Mistral-7B)")
        print("=" * 70)
        print(f"🚀 GPU Available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        print(f"🔧 PyTorch CUDA: {torch.cuda.is_available()}")
    else:
        print("🔒 FIXED Differential Privacy Author Name Perturbation (Mistral-7B)")
        print("=" * 70)
        print("⚠️  GPU NOT Available - using CPU (will be very slow!)")
    
    print(f"🔧 Configuration:")
    print(f"   Model: {args.model_path}")
    print(f"   Privacy ε: {args.epsilon}")
    print(f"   Failure probability δ: {args.delta}")
    print(f"   Input: {args.train_json_path}")
    print(f"   Author mapping: {args.author_mapping_path}")
    print(f"   Output: {args.output_path}")
    print("=" * 70)
    
    # Check input files
    if not os.path.exists(args.train_json_path):
        print(f"❌ Training file not found: {args.train_json_path}")
        return 1
    
    if not os.path.exists(args.author_mapping_path):
        print(f"❌ Author mapping not found: {args.author_mapping_path}")
        return 1
    
    # Initialize perturbator
    perturbator = None
    try:
        perturbator = AuthorDatasetPerturbator(
            model_path=args.model_path,
            epsilon=args.epsilon,
            delta=args.delta,
            verbose=True
        )
        
        # Show final device being used
        device_info = perturbator.embedding_extractor.device
        if device_info.type == 'cuda':
            print(f"✅ Using GPU for Mistral-7B embeddings")
        else:
            print(f"⚠️  Using CPU - this will be much slower!")
        print("=" * 70)
        
        # Load author mapping
        print("📥 Loading and validating author mapping...")
        author_mapping = perturbator.load_author_mapping(args.author_mapping_path)
        
        # Validate that the mapping matches expected format
        expected_indices = {
            "1", "6", "7", "8", "11", "22", "23", "24", "26", "28", "35", "39", "40", 
            "50", "55", "56", "57", "59", "62", "67", "70", "71", "86", "87", "88", 
            "91", "97", "107", "108", "114", "117", "129", "137", "139", "143", "150", 
            "151", "154", "163", "166", "168", "173", "181", "182", "185", "188", 
            "189", "190", "191", "197"
        }
        
        loaded_indices = set(author_mapping.keys())
        if loaded_indices == expected_indices:
            print("✅ Author mapping matches expected format")
        else:
            missing = expected_indices - loaded_indices
            extra = loaded_indices - expected_indices
            if missing:
                print(f"⚠️  Missing author indices: {sorted(missing)}")
            if extra:
                print(f"⚠️  Extra author indices: {sorted(extra)}")
        
        # Build author embedding space
        print("\n🏗️  Building author embedding space...")
        perturbator.build_author_embedding_space(author_mapping)
        
        # Process dataset
        print("\n📖 Processing dataset...")
        stats = perturbator.process_dataset(
            train_json_path=args.train_json_path,
            author_mapping=author_mapping,
            output_path=args.output_path
        )
        
        print(f"\n🎉 Success! DP-perturbed dataset saved to: {args.output_path}")
        
        # Save processing metadata
        metadata = {
            "epsilon": args.epsilon,
            "delta": args.delta,
            "sensitivity": perturbator.dp_perturbator.sensitivity,
            "noise_scale": perturbator.dp_perturbator.noise_scale,
            "model_path": args.model_path,
            "model_type": "Mistral-7B-Instruct-v0.2",
            "total_authors": len(set(author_mapping.values())),
            "author_indices_count": len(author_mapping),
            "processing_stats": stats,
            "privacy_guarantee": f"({args.epsilon}, {args.delta})-DP",
            "mechanism": "Gaussian mechanism with L2 sensitivity",
            "input_format": "JSON",
            "qa_pairs_processed": sum(stats.values()),
            "independent_perturbation": True,
            "note": "Each QA pair perturbed independently with FIXED implementation",
            "fixes_applied": [
                "Thread-safe embedding extraction using threading.local",
                "Robust hook setup with systematic fallback locations",
                "Better memory management with batch processing",
                "Improved error handling for embedding extraction",
                "Fixed text replacement logic for different name structures",
                "Better random state management with isolated RandomState",
                "Comprehensive validation for embedding operations",
                "Enhanced GPU memory cleanup",
                "Better handling of zero embeddings and edge cases"
            ],
            "author_mapping_sample": {
                "1": "Chukwu Akabueze",
                "6": "Elliot Patrick Benson", 
                "88": "author",
                "total_entries": len(author_mapping)
            }
        }
        
        metadata_path = args.output_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadata saved to: {metadata_path}")
        
        # Additional validation of output
        print("\n🔍 Validating output...")
        try:
            with open(args.output_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            
            print(f"✅ Output file is valid JSON with {len(output_data)} entries")
            
            # Check a few samples
            if output_data:
                sample = output_data[0]
                required_new_fields = ['original_author_name', 'perturbed_author_name', 'perturbation_applied']
                present_fields = [field for field in required_new_fields if field in sample]
                print(f"✅ New metadata fields present: {present_fields}")
                
                # Show some perturbation examples
                perturbation_examples = []
                for qa in output_data[:10]:
                    if qa.get('perturbation_applied', False):
                        orig = qa.get('original_author_name', 'N/A')
                        pert = qa.get('perturbed_author_name', 'N/A')
                        perturbation_examples.append(f"{orig} → {pert}")
                
                if perturbation_examples:
                    print(f"✅ Sample perturbations: {perturbation_examples[:3]}")
                else:
                    print("ℹ️  No perturbations found in first 10 entries")
        
        except Exception as e:
            print(f"⚠️  Output validation error: {e}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if perturbator:
            perturbator.cleanup()

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)