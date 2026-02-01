#!/usr/bin/env python3
"""
KNN-LM Generation with FINE-TUNED EMBEDDINGS: PERMANENT AUTHOR REMOVAL VERSION
LM: Pre-trained Mistral 7B (unchanged)
Embeddings: Fine-tuned Mistral 7B (for both datastore building and evaluation)
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import time
from typing import List, Tuple, Dict, Set
import sys
import faiss
import argparse
import os
import random

class AuthorRemovalKNNLMGenerator:
    """KNN-LM generator with FINE-TUNED EMBEDDINGS and PERMANENT AUTHOR REMOVAL"""
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 adapter_path: str = "./model_checkpoints/mistral_tofu_lora_fixed_20250722_133126",
                 batch_size: int = 256, k: int = 10, upper_lambda: float = 1.0, 
                 lower_lambda: float = 0.0, distance_threshold: float = 0.1,
                 removed_entity_percentage: float = 0.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.k = k
        self.upper_lambda = upper_lambda
        self.lower_lambda = lower_lambda
        self.distance_threshold = distance_threshold
        self.removed_entity_percentage = removed_entity_percentage
        
        print("⚡ KNN-LM GENERATOR (FINE-TUNED EMBEDDINGS + PERMANENT AUTHOR REMOVAL)")
        print("=" * 70)
        print(f"Base Model: {model_path}")
        print(f"Fine-tuned Adapter: {adapter_path}")
        print(f"Device: {self.device}")
        print(f"K neighbors: {k}")
        print(f"Upper lambda (near): {upper_lambda}")
        print(f"Lower lambda (far): {lower_lambda}")
        print(f"Distance threshold: {distance_threshold}")
        print(f"Removed entity percentage: {removed_entity_percentage*100}%")
        print(f"🔥 LM: Pre-trained | Embeddings: Fine-tuned")
        print(f"TASK: Combined1NN (DYNAMIC λ) + PERMANENT AUTHOR REMOVAL")
        sys.stdout.flush()
        
        # Set deterministic mode
        self._set_deterministic()
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        sys.stdout.flush()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load PRE-TRAINED model for LM generation
        print("🤖 Loading PRE-TRAINED model (for LM generation)...")
        sys.stdout.flush()
        self.lm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.lm_model.eval()
        for param in self.lm_model.parameters():
            param.requires_grad = False
        
        # Load FINE-TUNED model for embeddings
        print("🎯 Loading FINE-TUNED model (for embeddings)...")
        sys.stdout.flush()
        base_embedding_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        
        # Load fine-tuned adapter for embeddings
        print("🎯 Loading fine-tuned LoRA adapter for embeddings...")
        sys.stdout.flush()
        self.embedding_model = PeftModel.from_pretrained(base_embedding_model, adapter_path)
        self.embedding_model.eval()
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        
        # Setup hook for embedding extraction on FINE-TUNED model
        self.embeddings = []
        self.hook_handles = []
        self._setup_hook()
        
        # FAISS datastore
        self.faiss_index = None
        self.datastore_contexts = []
        self.datastore_targets = []
        self.datastore_author_indices = []
        
        # Track removed authors
        self.removed_authors: Set[int] = set()
        
        print("✅ Initialization complete")
        print(f"   🤖 LM Model: Pre-trained {model_path}")
        print(f"   🎯 Embedding Model: Fine-tuned {adapter_path}")
        sys.stdout.flush()
    
    def _set_deterministic(self):
        """Set deterministic mode"""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_hook(self):
        """Setup embedding hook on FINE-TUNED model"""
        print("🔗 Setting up embedding hook on FINE-TUNED model...")
        sys.stdout.flush()
        
        try:
            target_module = self.embedding_model.model.model.layers[-1].post_attention_layernorm
            print("✅ Found hook location: embedding_model.model.layers[-1].post_attention_layernorm")
        except:
            try:
                target_module = self.embedding_model.model.layers[-1].post_attention_layernorm
                print("✅ Found hook location: embedding_model.layers[-1].post_attention_layernorm")
            except Exception as e:
                print(f"❌ Hook setup failed: {e}")
                raise
        
        def hook_fn(module, input, output):
            try:
                if output is not None:
                    self.embeddings.append(output.detach().cpu())
            except Exception as e:
                print(f"Hook error: {e}")
        
        handle = target_module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)
        print("✅ Hook registered on fine-tuned model")
        sys.stdout.flush()
    
    def get_embedding_single(self, text: str) -> np.ndarray:
        """Extract embedding for single text using FINE-TUNED model"""
        self._set_deterministic()
        
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            self.embeddings.clear()
            
            with torch.amp.autocast('cuda'):
                outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            
            if self.embeddings:
                batch_embeddings = self.embeddings[0]
                seq_length = attention_mask.sum(dim=1) - 1
                embedding = batch_embeddings[0, seq_length.item()]
                
                embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
                return embedding.squeeze().numpy()
            else:
                raise RuntimeError("No embedding extracted from hook")
    
    def extract_embeddings_SMART_BATCH(self, contexts: List[str]) -> np.ndarray:
        """SMART BATCHING: Extract embeddings using FINE-TUNED model"""
        print(f"⚡ SMART BATCH processing {len(contexts)} contexts with FINE-TUNED embeddings")
        sys.stdout.flush()
        
        all_embeddings = []
        num_batches = (len(contexts) + self.batch_size - 1) // self.batch_size
        print(f"📊 Total batches: {num_batches}")
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(contexts))
            batch_contexts = contexts[start_idx:end_idx]
            
            if batch_idx % 20 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start_time
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Batch {batch_idx + 1}/{num_batches} ({progress:.1f}%)")
                sys.stdout.flush()
            
            try:
                with torch.no_grad():
                    self._set_deterministic()
                    
                    batch_embeddings = []
                    batch_input_ids = []
                    batch_attention_masks = []
                    batch_seq_lengths = []
                    
                    for context in batch_contexts:
                        encoding = self.tokenizer(
                            context,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        )
                        
                        input_ids = encoding['input_ids'].squeeze()
                        attention_mask = encoding['attention_mask'].squeeze()
                        
                        batch_input_ids.append(input_ids)
                        batch_attention_masks.append(attention_mask)
                        batch_seq_lengths.append(attention_mask.sum().item() - 1)
                    
                    max_len = max(len(ids) for ids in batch_input_ids)
                    
                    padded_input_ids = []
                    padded_attention_masks = []
                    
                    for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks):
                        pad_length = max_len - len(input_ids)
                        
                        if pad_length > 0:
                            padded_ids = torch.cat([
                                input_ids,
                                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                            ])
                            padded_mask = torch.cat([
                                attention_mask,
                                torch.zeros(pad_length, dtype=attention_mask.dtype)
                            ])
                        else:
                            padded_ids = input_ids
                            padded_mask = attention_mask
                        
                        padded_input_ids.append(padded_ids)
                        padded_attention_masks.append(padded_mask)
                    
                    batch_input_ids_tensor = torch.stack(padded_input_ids).to(self.device)
                    batch_attention_masks_tensor = torch.stack(padded_attention_masks).to(self.device)
                    
                    self.embeddings.clear()
                    
                    with torch.amp.autocast('cuda'):
                        outputs = self.embedding_model(
                            input_ids=batch_input_ids_tensor,
                            attention_mask=batch_attention_masks_tensor
                        )
                    
                    if self.embeddings:
                        hook_output = self.embeddings[0]
                        
                        for i, seq_length in enumerate(batch_seq_lengths):
                            embedding = hook_output[i, seq_length]
                            embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
                            batch_embeddings.append(embedding.squeeze().cpu().numpy())
                        
                        all_embeddings.extend(batch_embeddings)
                    else:
                        batch_size = len(batch_contexts)
                        zeros = [np.zeros(4096) for _ in range(batch_size)]
                        all_embeddings.extend(zeros)
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n❌ GPU OOM with batch_size={self.batch_size}!")
                    raise
                else:
                    raise
        
        total_time = time.time() - start_time
        print(f"✅ SMART BATCH complete with FINE-TUNED embeddings! Time: {total_time:.1f}s")
        sys.stdout.flush()
        
        return np.array(all_embeddings)
    
    def create_context_target_pairs(self, question: str, answer: str, author_index: int) -> List[Tuple[str, int, int]]:
        """Create progressive context-target pairs with author_index"""
        
        q_encoding = self.tokenizer(question, add_special_tokens=False, return_tensors='pt')
        a_encoding = self.tokenizer(answer, add_special_tokens=False, return_tensors='pt')
        
        q_ids = q_encoding['input_ids'].squeeze().tolist()
        a_ids = a_encoding['input_ids'].squeeze().tolist()
        
        if not isinstance(q_ids, list):
            q_ids = [q_ids]
        if not isinstance(a_ids, list):
            a_ids = [a_ids]
        
        full_ids = q_ids + a_ids
        pairs = []
        
        for i in range(len(q_ids), len(full_ids)):
            context_ids = full_ids[:i]
            target_id = full_ids[i]
            
            context_text = self.tokenizer.decode(context_ids, skip_special_tokens=True)
            pairs.append((context_text, target_id, author_index))
        
        return pairs
    
    def select_authors_to_remove(self, test_data: List[dict]) -> Set[int]:
        """Randomly select authors to remove based on percentage"""
        print(f"\n🎯 SELECTING AUTHORS TO REMOVE ({self.removed_entity_percentage*100}%)")
        print("-" * 50)
        
        # Get all unique author indices from test data
        all_test_authors = set()
        for qa in test_data:
            author_index = qa.get('author_index')
            if author_index is not None:
                all_test_authors.add(author_index)
        
        all_test_authors = sorted(list(all_test_authors))
        total_authors = len(all_test_authors)
        
        print(f"Total unique authors in test set: {total_authors}")
        print(f"Authors: {all_test_authors}")
        
        # Calculate how many to remove
        num_to_remove = int(total_authors * self.removed_entity_percentage)
        
        print(f"Number of authors to remove: {num_to_remove}")
        
        if num_to_remove == 0:
            print("✅ No authors selected for removal")
            return set()
        
        # Randomly select authors to remove
        random.seed(42)  # For reproducibility
        removed_authors = set(random.sample(all_test_authors, num_to_remove))
        
        print(f"Selected authors for removal: {sorted(list(removed_authors))}")
        print(f"Retained authors: {sorted([a for a in all_test_authors if a not in removed_authors])}")
        
        return removed_authors
    
    def build_datastore_with_permanent_removal(self, train_data: List[dict], removed_authors: Set[int]):
        """Build datastore with permanent author removal using author_index"""
        print(f"\n🔨 BUILDING DATASTORE WITH PERMANENT AUTHOR REMOVAL")
        print("-" * 70)
        print(f"Authors to permanently remove: {sorted(list(removed_authors))}")
        sys.stdout.flush()
        
        start_time = time.time()
        
        all_contexts = []
        all_targets = []
        all_author_indices = []
        
        removed_qa_count = 0
        retained_qa_count = 0
        removed_pairs_count = 0
        
        for qa_idx, qa in enumerate(train_data):
            question = qa['question']
            answer = qa['answer']
            author_index = qa.get('author_index', qa_idx)
            
            if qa_idx % 500 == 0 or qa_idx == len(train_data) - 1:
                progress_pct = (qa_idx + 1) / len(train_data) * 100
                print(f"Processing Q&A {qa_idx + 1}/{len(train_data)} ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            # Skip if author is in removal list
            if author_index in removed_authors:
                removed_qa_count += 1
                pairs = self.create_context_target_pairs(question, answer, author_index)
                removed_pairs_count += len(pairs)
                continue
            
            retained_qa_count += 1
            pairs = self.create_context_target_pairs(question, answer, author_index)
            
            for context, target, auth_idx in pairs:
                all_contexts.append(context)
                all_targets.append(target)
                all_author_indices.append(auth_idx)
        
        pairs_time = time.time()
        print(f"✅ Datastore statistics:")
        print(f"   Total train Q&A pairs: {len(train_data)}")
        print(f"   Removed Q&A pairs: {removed_qa_count}")
        print(f"   Retained Q&A pairs: {retained_qa_count}")
        print(f"   Removed context-target pairs: {removed_pairs_count}")
        print(f"   Retained context-target pairs: {len(all_contexts)}")
        print(f"   Context-target pair creation time: {pairs_time - start_time:.1f}s")
        sys.stdout.flush()
        
        if len(all_contexts) == 0:
            raise ValueError("No contexts remain after author removal!")
        
        # Extract embeddings using FINE-TUNED model
        embeddings = self.extract_embeddings_SMART_BATCH(all_contexts)
        
        embedding_time = time.time()
        print(f"⏱️ FINE-TUNED embedding time: {embedding_time - pairs_time:.1f}s")
        sys.stdout.flush()
        
        print("🔍 Building FAISS index...")
        sys.stdout.flush()
        
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        self.datastore_contexts = all_contexts
        self.datastore_targets = all_targets
        self.datastore_author_indices = all_author_indices
        
        total_time = time.time() - start_time
        print(f"\n✅ Datastore built with PERMANENT AUTHOR REMOVAL: {len(all_contexts)} pairs, {total_time:.1f}s")
        print(f"   🗑️ Permanently removed {removed_qa_count} Q&A pairs from {len(removed_authors)} authors")
        sys.stdout.flush()
    
    def generate_combined_1nn_dynamic(self, question: str, max_tokens: int = 75) -> str:
        """TASK 2b: Generate using Combined LM + 1NN with DYNAMIC LAMBDA per token"""
        print(f"⚖️ COMBINED LM+1NN GENERATION (DYNAMIC λ) - GREEDY:")
        print(f"Question: {question}")
        print(f"Upper λ: {self.upper_lambda} (distance < {self.distance_threshold})")
        print(f"Lower λ: {self.lower_lambda} (distance >= {self.distance_threshold})")
        print(f"Current datastore size: {len(self.datastore_contexts)}")
        
        # Start with contexts as text
        lm_prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        knn_context = question  # Start with question as text
        generated_tokens = []
        
        for step in range(max_tokens):
            # Get LM distribution using PRE-TRAINED model
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                current_lm_prompt = f"{lm_prompt} {generated_text}"
            else:
                current_lm_prompt = lm_prompt
                
            with torch.no_grad():
                encoding = self.tokenizer(current_lm_prompt, return_tensors='pt', truncation=True, max_length=1024)
                input_ids = encoding['input_ids'].to(self.device)
                
                # Use PRE-TRAINED model for LM probabilities
                outputs = self.lm_model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
                lm_probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            # Get 1NN distribution using FINE-TUNED embeddings
            query_embedding = self.get_embedding_single(knn_context)  # Uses fine-tuned model
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 1  # Only 1 neighbor
            )
            
            # Calculate dynamic lambda based on 1NN distance
            shortest_distance = 1.0 - similarities[0][0]  # Distance from 1NN
            
            # Dynamic lambda assignment
            if shortest_distance < self.distance_threshold:
                dynamic_lambda = self.upper_lambda  # Close neighbor -> high KNN weight
            else:
                dynamic_lambda = self.lower_lambda  # Far neighbor -> low KNN weight
            
            # Get the 1NN target token
            target_token = self.datastore_targets[indices[0][0]]
            
            # Create 1NN distribution: prob=1 for target token, prob=0 for all others
            vocab_size = len(self.tokenizer)
            knn_probs = np.zeros(vocab_size)
            if 0 <= target_token < vocab_size:
                knn_probs[target_token] = 1.0  # Probability 1 for 1NN token
            
            # Combine distributions with DYNAMIC lambda
            combined_probs = (1.0 - dynamic_lambda) * lm_probs + dynamic_lambda * knn_probs
            combined_probs = combined_probs / combined_probs.sum()
            
            # Use GREEDY DETERMINISTIC selection
            next_token_id = np.argmax(combined_probs)
            
            if step < 5:  # Show first few steps
                next_token_text = self.tokenizer.decode([next_token_id])
                lm_prob = lm_probs[next_token_id]
                knn_prob = knn_probs[next_token_id]
                combined_prob = combined_probs[next_token_id]
                print(f"  Step {step}: '{next_token_text}' | Dist:{shortest_distance:.6f} λ:{dynamic_lambda:.2f}")
                print(f"           LM:{lm_prob:.4f} 1NN:{knn_prob:.4f} Combined:{combined_prob:.4f} [GREEDY]")
            
            # Stop conditions
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            # Add to generation
            generated_tokens.append(next_token_id)
            
            # Update KNN context
            question_tokens = self.tokenizer(question, add_special_tokens=False)['input_ids']
            full_tokens = question_tokens + generated_tokens
            knn_context = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
            
            # Stop on sentence end
            if len(generated_tokens) > 15:
                recent_text = self.tokenizer.decode([next_token_id])
                if any(punct in recent_text for punct in ['.', '!', '?']):
                    break
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"✅ Combined LM+1NN Answer: '{generated_text.strip()}'")
        return generated_text.strip()
    
    def evaluate_on_questions_split(self, test_data: List[dict], removed_authors: Set[int], 
                                   removed_output_file: str, retained_output_file: str):
        """Evaluate and split results based on author removal"""
        print(f"\n🧪 EVALUATION WITH AUTHOR-BASED RESULT SPLITTING")
        print("=" * 80)
        print(f"Removed authors: {sorted(list(removed_authors))}")
        print(f"Output for removed authors: {removed_output_file}")
        print(f"Output for retained authors: {retained_output_file}")
        sys.stdout.flush()
        
        removed_results = []
        retained_results = []
        
        removed_count = 0
        retained_count = 0
        
        for qa_idx, qa in enumerate(test_data):
            question = qa['question']
            ground_truth = qa['answer']
            author_index = qa.get('author_index', None)
            
            print(f"\n{'='*10} QUESTION {qa_idx + 1} {'='*10}")
            print(f"Q: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Author Index: {author_index}")
            
            # Generate answer using current datastore (with permanent removals)
            generated_answer = self.generate_combined_1nn_dynamic(question)
            
            # Create result entry
            result_entry = qa.copy()
            result_entry['answer'] = generated_answer
            
            # Split based on whether author was removed
            if author_index in removed_authors:
                removed_results.append(result_entry)
                removed_count += 1
                print(f"✅ Added to REMOVED authors results")
            else:
                retained_results.append(result_entry)
                retained_count += 1
                print(f"✅ Added to RETAINED authors results")
            
            print(f"Generated Answer: {generated_answer}")
            print(f"{'='*50}")
        
        # Save results to separate JSON files
        print(f"\n💾 Saving results...")
        print(f"   Removed authors results: {removed_count} entries → {removed_output_file}")
        print(f"   Retained authors results: {retained_count} entries → {retained_output_file}")
        
        # Create output directories
        os.makedirs(os.path.dirname(removed_output_file), exist_ok=True)
        os.makedirs(os.path.dirname(retained_output_file), exist_ok=True)
        
        # Save removed authors results
        with open(removed_output_file, 'w', encoding='utf-8') as f:
            json.dump(removed_results, f, ensure_ascii=False, indent=2)
        
        # Save retained authors results
        with open(retained_output_file, 'w', encoding='utf-8') as f:
            json.dump(retained_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Results saved successfully!")
        
        return removed_results, retained_results
    
    def load_qa_data(self, filepath: str):
        """Load Q&A data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} Q&A pairs from {filepath}")
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            return []
    
    def run_complete_pipeline(self, train_file: str, test_file: str, 
                            removed_output_file: str, retained_output_file: str):
        """Run complete pipeline with permanent author removal"""
        print(f"\n🎯 KNN-LM PIPELINE WITH PERMANENT AUTHOR REMOVAL")
        print(f"📁 Train: {train_file}")
        print(f"📁 Test: {test_file}")
        print(f"🗑️ Removed authors output: {removed_output_file}")
        print(f"✅ Retained authors output: {retained_output_file}")
        print(f"📊 Removal percentage: {self.removed_entity_percentage*100}%")
        
        # Load data
        train_data = self.load_qa_data(train_file)
        test_data = self.load_qa_data(test_file)
        
        if not train_data or not test_data:
            print("❌ Missing data files")
            return
        
        # Select authors to remove based on test set
        removed_authors = self.select_authors_to_remove(test_data)
        self.removed_authors = removed_authors
        
        # Build datastore with permanent author removal
        self.build_datastore_with_permanent_removal(train_data, removed_authors)
        
        # Evaluate and split results
        removed_results, retained_results = self.evaluate_on_questions_split(
            test_data, removed_authors, removed_output_file, retained_output_file
        )
        
        print(f"\n🎯 PIPELINE COMPLETE!")
        print(f"   Authors permanently removed: {len(removed_authors)} ({self.removed_entity_percentage*100}%)")
        print(f"   Removed authors results: {len(removed_results)} Q&A pairs")
        print(f"   Retained authors results: {len(retained_results)} Q&A pairs")
        print(f"   🤖 LM: Pre-trained | 🎯 Embeddings: Fine-tuned")
        print(f"   🗑️ Permanent author removal: Enabled")
        
        return removed_results, retained_results
    
    def cleanup(self):
        """Cleanup"""
        for handle in self.hook_handles:
            handle.remove()
        if hasattr(self, 'lm_model'):
            del self.lm_model
        if hasattr(self, 'embedding_model'):
            del self.embedding_model
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--removed-output-file', required=True, 
                       help='Output JSON file for removed authors')
    parser.add_argument('--retained-output-file', required=True,
                       help='Output JSON file for retained authors')
    parser.add_argument('--adapter-path', default='./model_checkpoints/mistral_tofu_lora_fixed_20250722_133126', 
                       help='Path to fine-tuned LoRA adapter')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--upper-lambda', type=float, default=1.0, 
                       help='Lambda weight when distance < threshold')
    parser.add_argument('--lower-lambda', type=float, default=0.0,
                       help='Lambda weight when distance >= threshold')
    parser.add_argument('--distance-threshold', type=float, default=0.1,
                       help='Distance threshold for dynamic lambda assignment')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--removed-entity-percentage', type=float, default=0.05,
                       help='Percentage of authors to remove (0.05 = 5%)')
    
    args = parser.parse_args()
    
    generator = None
    try:
        generator = AuthorRemovalKNNLMGenerator(
            adapter_path=args.adapter_path,
            batch_size=args.batch_size,
            k=args.k,
            upper_lambda=args.upper_lambda,
            lower_lambda=args.lower_lambda,
            distance_threshold=args.distance_threshold,
            removed_entity_percentage=args.removed_entity_percentage
        )
        
        results = generator.run_complete_pipeline(
            args.train_file, args.test_file, 
            args.removed_output_file, args.retained_output_file
        )
        print(f"\n✅ PERMANENT AUTHOR REMOVAL TASK COMPLETED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if generator:
            generator.cleanup()

if __name__ == "__main__":
    main()