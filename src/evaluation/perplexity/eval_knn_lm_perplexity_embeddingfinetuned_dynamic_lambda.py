#!/usr/bin/env python3
"""
KNN-LM Perplexity Evaluation with FINE-TUNED EMBEDDINGS: DYNAMIC LAMBDA VERSION
LM: Pre-trained Mistral 7B (unchanged)
Embeddings: Fine-tuned Mistral 7B (for both datastore building and evaluation)
🔧 UPDATED VERSION: Uses same perplexity evaluation approach as LM-only evaluator
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import time
from typing import List, Tuple, Dict
import sys
import faiss
import argparse
import os
import math

class KNNLMPerplexityEvaluatorFineTuned:
    """KNN-LM perplexity evaluator with FINE-TUNED EMBEDDINGS and DYNAMIC LAMBDA per token"""
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 adapter_path: str = "./model_checkpoints/mistral_tofu_lora_fixed_20250722_133126",
                 batch_size: int = 256, k: int = 10, upper_lambda: float = 1.0, 
                 lower_lambda: float = 0.0, distance_threshold: float = 0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.k = k
        self.upper_lambda = upper_lambda  # High weight to KNN when distance < threshold
        self.lower_lambda = lower_lambda  # Low weight to KNN when distance >= threshold
        self.distance_threshold = distance_threshold
        
        print("⚡ KNN-LM PERPLEXITY EVALUATOR (FINE-TUNED EMBEDDINGS + UPDATED PERPLEXITY EVALUATION)")
        print("=" * 70)
        print(f"Base Model: {model_path}")
        print(f"Fine-tuned Adapter: {adapter_path}")
        print(f"Device: {self.device}")
        print(f"K neighbors: {k}")
        print(f"Upper lambda (near): {upper_lambda}")
        print(f"Lower lambda (far): {lower_lambda}")
        print(f"Distance threshold: {distance_threshold}")
        print(f"🔥 LM: Pre-trained | Embeddings: Fine-tuned")
        print(f"TASK: Perplexity Evaluation with Combined LM+1NN (DYNAMIC λ)")
        print("🔧 UPDATED: Same perplexity evaluation as LM-only baseline")
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
        
        # FAISS datastore (will be loaded later)
        self.faiss_index = None
        self.datastore_contexts = []
        self.datastore_targets = []
        
        print("✅ Initialization complete")
        print(f"   🤖 LM Model: Pre-trained {model_path}")
        print(f"   🎯 Embedding Model: Fine-tuned {adapter_path}")
        sys.stdout.flush()
    
    def _set_deterministic(self):
        """Set deterministic mode"""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        import random
        import numpy as np
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
        self._set_deterministic()  # Ensure deterministic extraction
        
        with torch.no_grad():
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Clear previous embeddings
            self.embeddings.clear()
            
            # Forward pass through FINE-TUNED model
            with torch.amp.autocast('cuda'):
                outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extract embedding from hook
            if self.embeddings:
                batch_embeddings = self.embeddings[0]  # [1, seq_len, hidden_dim]
                seq_length = attention_mask.sum(dim=1) - 1  # Last token position
                embedding = batch_embeddings[0, seq_length.item()]
                
                # L2 normalize
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
                    
                    # Forward pass through FINE-TUNED model
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
    
    def create_context_target_pairs(self, question: str, answer: str) -> List[Tuple[str, int]]:
        """Create progressive context-target pairs - EXACT same as working version"""
        
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
            pairs.append((context_text, target_id))
        
        return pairs
    
    def build_datastore(self, qa_pairs: List[dict]):
        """Build datastore using FINE-TUNED embeddings"""
        print("\n🔨 BUILDING DATASTORE (FINE-TUNED EMBEDDINGS + SMART BATCHING + FAISS)")
        print("-" * 70)
        sys.stdout.flush()
        
        start_time = time.time()
        
        all_contexts = []
        all_targets = []
        
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa['question']
            answer = qa['answer']
            
            if qa_idx % 500 == 0 or qa_idx == len(qa_pairs) - 1:
                progress_pct = (qa_idx + 1) / len(qa_pairs) * 100
                print(f"Processing Q&A {qa_idx + 1}/{len(qa_pairs)} ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            pairs = self.create_context_target_pairs(question, answer)
            
            for context, target in pairs:
                all_contexts.append(context)
                all_targets.append(target)
        
        pairs_time = time.time()
        print(f"✅ Created {len(all_contexts)} context-target pairs in {pairs_time - start_time:.1f}s")
        sys.stdout.flush()
        
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
        
        total_time = time.time() - start_time
        print(f"\n✅ Datastore built with FINE-TUNED embeddings: {len(all_contexts)} pairs, {total_time:.1f}s")
        sys.stdout.flush()
    
    def calculate_knn_lm_perplexity(self, question: str, ground_truth_answer: str) -> float:
        """
        🔧 UPDATED VERSION: Uses EXACT same perplexity evaluation approach as LM-only evaluator
        When lambda=0.0, this should give IDENTICAL results to LM-only baseline
        """
        print(f"⚖️ COMBINED LM+1NN PERPLEXITY EVALUATION (UPDATED PERPLEXITY EVALUATION):")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth_answer}")
        print(f"Upper λ: {self.upper_lambda} (distance < {self.distance_threshold})")
        print(f"Lower λ: {self.lower_lambda} (distance >= {self.distance_threshold})")
        
        # 🔧 UPDATED: Use EXACT same approach as LM-only evaluator
        prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        full_sequence = f"{prompt} {ground_truth_answer}"
        
        # Tokenize the full sequence at once (like LM-only - avoids alignment issues)
        full_encoding = self.tokenizer(full_sequence, return_tensors='pt', truncation=True, max_length=1024)
        full_input_ids = full_encoding['input_ids'].to(self.device)
        attention_mask = full_encoding['attention_mask'].to(self.device)
        
        # Tokenize just the prompt to find where answer starts (like LM-only)
        prompt_encoding = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        # 🔧 UPDATED: Add same [/INST] token detection logic as LM-only
        # Better boundary detection - verify tokenization consistency (same logic as LM-only)
        # Check if the prompt portion matches when tokenized separately vs as part of full sequence
        prompt_from_full = full_input_ids[0, :prompt_length]
        prompt_separate = prompt_encoding['input_ids'][0].to(self.device)
        
        # Same fallback logic as LM-only, with FIXED [/INST] token detection
        if not torch.equal(prompt_from_full, prompt_separate):
            # Handle multi-token [/INST] sequence
            inst_close_tokens = self.tokenizer.encode("[/INST]", add_special_tokens=False)
            
            # Convert to tensor for searching
            full_ids = full_input_ids[0]
            inst_close_tensor = torch.tensor(inst_close_tokens, device=self.device)
            
            # Find the position where [/INST] sequence ends
            found_position = None
            for i in range(len(full_ids) - len(inst_close_tokens) + 1):
                if torch.equal(full_ids[i:i+len(inst_close_tokens)], inst_close_tensor):
                    found_position = i + len(inst_close_tokens)  # Position after [/INST]
                    break
            
            if found_position is not None:
                prompt_length = found_position
            # else: keep original prompt_length as fallback
        
        # Verify we have answer tokens (like LM-only)
        if full_input_ids.shape[1] <= prompt_length:
            return float('inf')  # No answer tokens
        
        # Get answer tokens (same calculation as LM-only)
        answer_start_idx = prompt_length
        answer_end_idx = full_input_ids.shape[1]
        answer_length = answer_end_idx - answer_start_idx
        
        if answer_length == 0:
            return float('inf')
        
        # 🔧 UPDATED: Add same truncation check as LM-only
        # Check for truncation (same check as LM-only)
        if attention_mask[0, -1] == 0:  # Last token is masked (truncated)
            print(f"   Warning: Sequence may be truncated (length: {full_input_ids.shape[1]})")
        
        # 🔧 UPDATED: Calculate perplexity using single forward pass for LM (like LM-only)
        with torch.no_grad():
            # Use PRE-TRAINED model for LM probabilities (single forward pass)
            outputs = self.lm_model(input_ids=full_input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # 🔧 UPDATED: Use log_softmax for numerical stability (same as LM-only)
            log_probs = F.log_softmax(logits, dim=-1)
            
            total_log_prob = 0.0
            num_tokens = 0
            
            # For each answer token, get its probability given previous context (like LM-only)
            for i in range(answer_length):
                # Position in sequence: we want to predict token at (prompt_length + i)
                # using context up to (prompt_length + i - 1)
                target_pos = prompt_length + i
                context_pos = target_pos - 1
                
                if context_pos < 0 or target_pos >= full_input_ids.shape[1]:
                    continue
                
                # 🔧 UPDATED: Add same attention mask check as LM-only
                # Skip if attention mask indicates this token should be ignored (same check as LM-only)
                if attention_mask[0, target_pos] == 0:
                    continue
                
                # Get the target token ID
                target_token_id = full_input_ids[0, target_pos].item()
                
                # 🔧 UPDATED: Add bounds checking for token ID (same safety check as LM-only)
                if target_token_id < 0 or target_token_id >= log_probs.shape[1]:
                    continue
                
                # Get logits from context position for LM distribution
                context_logits = logits[context_pos]  # [vocab_size]
                
                # 🔧 UPDATED: Use log_softmax for LM probabilities (same as LM-only)
                lm_log_probs = F.log_softmax(context_logits, dim=-1).cpu().numpy()
                lm_probs = np.exp(lm_log_probs)  # Convert back to probabilities for KNN combination
                
                # 🔥 KEEP KNN LOGIC: Build context text for KNN embedding (up to current position)
                context_ids = full_input_ids[0, :target_pos]
                knn_context = self.tokenizer.decode(context_ids, skip_special_tokens=True)
                
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
                
                # Get probability of target token
                target_prob = combined_probs[target_token_id]
                
                # Show first few steps for debugging
                if i < 5:
                    target_token_text = self.tokenizer.decode([target_token_id])
                    lm_prob = lm_probs[target_token_id]
                    knn_prob = knn_probs[target_token_id]
                    print(f"  Step {i}: '{target_token_text}' | Dist:{shortest_distance:.6f} λ:{dynamic_lambda:.2f}")
                    print(f"           LM:{lm_prob:.4f} 1NN:{knn_prob:.4f} Combined:{target_prob:.4f}")
                
                # 🔧 UPDATED: Use same log probability handling as LM-only
                # Get log probability directly (same method as LM-only)
                if target_prob > 1e-10:  # Avoid log(0)
                    target_log_prob = math.log(target_prob)
                else:
                    target_log_prob = math.log(1e-10)  # Handle near-zero probabilities
                
                # 🔧 UPDATED: Add same finite value checking as LM-only
                # Proper error handling - skip non-finite values (same logic as LM-only)
                if math.isfinite(target_log_prob):
                    total_log_prob += target_log_prob
                    num_tokens += 1
        
        if num_tokens == 0:
            return float('inf')
        
        # 🔧 UPDATED: Calculate perplexity using same formula as LM-only
        # Calculate perplexity: exp(-1/N * sum(log(p_i))) (exact same formula as LM-only)
        avg_log_prob = total_log_prob / num_tokens
        perplexity = math.exp(-avg_log_prob)
        
        print(f"✅ KNN-LM Perplexity: {perplexity:.4f}")
        return perplexity
    
    def evaluate_perplexity_on_dataset(self, qa_pairs: List[dict]) -> Dict[str, float]:
        """Evaluate perplexity on the entire dataset"""
        print("\n🧪 KNN-LM PERPLEXITY EVALUATION WITH FINE-TUNED EMBEDDINGS")
        print("=" * 80)
        sys.stdout.flush()
        
        perplexities = []
        
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa['question']
            ground_truth = qa['answer']
            
            print(f"\n{'='*10} QUESTION {qa_idx + 1}/{len(qa_pairs)} {'='*10}")
            print(f"Q: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"{'='*50}")
            
            # 🔧 UPDATED: Add basic error handling for individual questions (same as LM-only)
            try:
                # Evaluate KNN-LM perplexity
                ppl = self.calculate_knn_lm_perplexity(question, ground_truth)
                perplexities.append(ppl)
                print(f"   KNN-LM perplexity: {ppl:.4f}")
            
            except Exception as e:
                print(f"   Error evaluating question {qa_idx + 1}: {e}")
                # Add inf value to maintain list alignment
                perplexities.append(float('inf'))
            
            print(f"{'='*80}")
            
            # 🔧 UPDATED: Add occasional memory cleanup to prevent OOM (same as LM-only)
            if (qa_idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 🔧 UPDATED: Handle potential inf values in averaging (same as LM-only)
        def safe_mean(values):
            finite_values = [v for v in values if math.isfinite(v)]
            return np.mean(finite_values) if finite_values else float('inf')
        
        avg_ppl = safe_mean(perplexities)
        
        results = {
            'knn_lm_perplexity': avg_ppl,
            'num_questions': len(qa_pairs)
        }
        
        return results
    
    def load_qa_data(self, filepath: str):
        """Load Q&A data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} Q&A pairs")
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            return []
        # 🔧 UPDATED: Add handling for JSON decode errors (same as LM-only)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return []
    
    def run_evaluation(self, train_file: str, test_file: str):
        """Run complete perplexity evaluation"""
        print(f"\n🎯 KNN-LM PERPLEXITY EVALUATION PIPELINE (FINE-TUNED EMBEDDINGS + UPDATED PERPLEXITY EVALUATION)")
        print(f"📁 Train: {train_file}")
        print(f"📁 Test: {test_file}")
        
        # Load data
        train_data = self.load_qa_data(train_file)
        test_data = self.load_qa_data(test_file)
        
        if not train_data or not test_data:
            print("❌ Missing data files")
            return None
        
        # Build datastore using fine-tuned embeddings
        self.build_datastore(train_data)
        
        # Evaluate perplexity
        results = self.evaluate_perplexity_on_dataset(test_data)
        
        # Print final results
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   📊 Dataset: {len(test_data)} questions")
        print(f"   ⚖️ KNN-LM Perplexity (Updated + Fine-tuned Embeddings): {results['knn_lm_perplexity']:.4f}")
        print(f"   🤖 LM: Pre-trained | 🎯 Embeddings: Fine-tuned")
        print(f"   🔧 Note: With λ=0.0, should match LM-only baseline exactly")
        
        return results
    
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
    parser.add_argument('--adapter-path', default='./model_checkpoints/mistral_tofu_lora_fixed_20250722_133126', 
                       help='Path to fine-tuned LoRA adapter')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--upper-lambda', type=float, default=1.0, 
                       help='Lambda weight when distance < threshold (high KNN weight)')
    parser.add_argument('--lower-lambda', type=float, default=0.0,
                       help='Lambda weight when distance >= threshold (low KNN weight)')
    parser.add_argument('--distance-threshold', type=float, default=0.1,
                       help='Distance threshold for dynamic lambda assignment')
    parser.add_argument('--batch-size', type=int, default=256)
    
    args = parser.parse_args()
    
    evaluator = None
    try:
        evaluator = KNNLMPerplexityEvaluatorFineTuned(
            adapter_path=args.adapter_path,
            batch_size=args.batch_size,
            k=args.k,
            upper_lambda=args.upper_lambda,
            lower_lambda=args.lower_lambda,
            distance_threshold=args.distance_threshold
        )
        
        results = evaluator.run_evaluation(args.train_file, args.test_file)
        
        if results:
            print("\n✅ KNN-LM PERPLEXITY EVALUATION WITH FINE-TUNED EMBEDDINGS (UPDATED) COMPLETED!")
            print(f"   ⚖️ KNN-LM Perplexity: {results['knn_lm_perplexity']:.4f}")
            print(f"   🤖 LM: Pre-trained | 🎯 Embeddings: Fine-tuned")
            print(f"   🔧 UPDATED: Uses same perplexity evaluation as LM-only baseline")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if evaluator:
            evaluator.cleanup()

if __name__ == "__main__":
    main()