#!/usr/bin/env python3
"""
KNN-LM Generation: FIVE TASK VERSION
Keep all existing tasks unchanged, add two new tasks:
- Task 1c: KNN-only (k neighbors with probability aggregation)  
- Task 2b: Combined LM + 1NN
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from typing import List, Tuple, Dict
import sys
import faiss
import argparse

class FiveTaskKNNLMGenerator:
    """KNN-LM generator with FIVE tasks: 1a, 1b-1NN, 1c-KNN, 2a-Combined, 2b-Combined1NN"""
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2", batch_size: int = 256, 
                 k: int = 10, lambda_weight: float = 0.25):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.k = k
        self.lambda_weight = lambda_weight
        
        print("⚡ KNN-LM GENERATOR (FIVE TASK VERSION)")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Device: {self.device}")
        print(f"K neighbors: {k}")
        print(f"Lambda weight: {lambda_weight}")
        print(f"FIVE TASKS: 1a-LM | 1b-1NN | 1c-KNN | 2a-Combined | 2b-Combined1NN")
        sys.stdout.flush()
        
        # Set deterministic mode
        self._set_deterministic()
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        sys.stdout.flush()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print("🤖 Loading model...")
        sys.stdout.flush()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False  # EXACT same as working version
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Setup hook for embedding extraction - EXACT same as working version
        self.embeddings = []
        self.hook_handles = []
        self._setup_hook()
        
        # FAISS datastore (will be loaded later)
        self.faiss_index = None
        self.datastore_contexts = []
        self.datastore_targets = []
        
        print("✅ Initialization complete")
        sys.stdout.flush()
    
    def _set_deterministic(self):
        """Set deterministic mode - EXACT same as working version"""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_hook(self):
        """Setup embedding hook - EXACT same as working version"""
        print("🔗 Setting up embedding hook...")
        sys.stdout.flush()
        
        try:
            target_module = self.model.model.layers[-1].post_attention_layernorm
            print("✅ Found hook location: model.layers[-1].post_attention_layernorm")
        except:
            try:
                target_module = self.model.layers[-1].post_attention_layernorm
                print("✅ Found hook location: layers[-1].post_attention_layernorm")
            except Exception as e:
                print(f"❌ Hook setup failed: {e}")
                raise
        
        def hook_fn(module, input, output):
            # EXACT same hook as working version
            try:
                if output is not None:
                    self.embeddings.append(output.detach().cpu())
            except Exception as e:
                print(f"Hook error: {e}")
        
        handle = target_module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)
        print("✅ Hook registered")
        sys.stdout.flush()
    
    def get_embedding_single(self, text: str) -> np.ndarray:
        """Extract embedding for single text - EXACT SAME as working version"""
        self._set_deterministic()  # Ensure deterministic extraction
        
        with torch.no_grad():
            # Tokenize - EXACT same parameters as working version
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=512,  # Keep 512 like working version
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Clear previous embeddings
            self.embeddings.clear()
            
            # Forward pass - EXACT same as working version
            with torch.amp.autocast('cuda'):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extract embedding from hook - EXACT same logic
            if self.embeddings:
                batch_embeddings = self.embeddings[0]  # [1, seq_len, hidden_dim]
                seq_length = attention_mask.sum(dim=1) - 1  # Last token position
                embedding = batch_embeddings[0, seq_length.item()]
                
                # L2 normalize - EXACT same
                embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
                return embedding.squeeze().numpy()
            else:
                raise RuntimeError("No embedding extracted from hook")
    
    # KEEP EXACT SAME SMART BATCHING FROM WORKING VERSION
    def extract_embeddings_SMART_BATCH(self, contexts: List[str]) -> np.ndarray:
        """SMART BATCHING: EXACT same as working version"""
        print(f"⚡ SMART BATCH processing {len(contexts)} contexts")
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
                        outputs = self.model(
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
        print(f"✅ SMART BATCH complete! Time: {total_time:.1f}s")
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
        """Build datastore - EXACT same as working version"""
        print("\n🔨 BUILDING DATASTORE (SMART BATCHING + FAISS)")
        print("-" * 60)
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
        
        embeddings = self.extract_embeddings_SMART_BATCH(all_contexts)
        
        embedding_time = time.time()
        print(f"⏱️ Embedding time: {embedding_time - pairs_time:.1f}s")
        sys.stdout.flush()
        
        print("🔍 Building FAISS index...")
        sys.stdout.flush()
        
        embedding_dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        self.datastore_contexts = all_contexts
        self.datastore_targets = all_targets
        
        total_time = time.time() - start_time
        print(f"\n✅ Datastore built: {len(all_contexts)} pairs, {total_time:.1f}s")
        sys.stdout.flush()
    
    # TASK 1a: LM-only generation - UNCHANGED
    def generate_lm_only(self, question: str, max_tokens: int = 75) -> str:
        """TASK 1a: Generate using LM only - UNCHANGED"""
        print(f"🤖 LM-ONLY GENERATION:")
        print(f"Question: {question}")
        
        # Start with prompt as text (similar to working version pattern)
        current_prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        generated_tokens = []
        
        for step in range(max_tokens):
            # Build current context - same pattern as working version
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                current_context = f"{current_prompt} {generated_text}"
            else:
                current_context = current_prompt
                
            with torch.no_grad():
                encoding = self.tokenizer(current_context, return_tensors='pt', truncation=True, max_length=1024)
                input_ids = encoding['input_ids'].to(self.device)
                
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                probs = F.softmax(logits, dim=-1)
                
                # GREEDY/DETERMINISTIC: Use argmax instead of sampling
                next_token_id = torch.argmax(probs).item()
                
                if step < 5:  # Show first few steps  
                    next_token_text = self.tokenizer.decode([next_token_id])
                    print(f"  Step {step}: '{next_token_text}' (prob: {probs[next_token_id]:.4f})")
                
                # Stop conditions
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                # Add to generation
                generated_tokens.append(next_token_id)
                
                # Stop on sentence end
                if len(generated_tokens) > 15:
                    recent_text = self.tokenizer.decode([next_token_id])
                    if any(punct in recent_text for punct in ['.', '!', '?']):
                        break
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"✅ LM Answer: '{generated_text.strip()}'")
        return generated_text.strip()
    
    # TASK 1b: 1NN-only generation - UNCHANGED (just renamed in prints)
    def generate_1nn_only(self, question: str, max_tokens: int = 30) -> str:
        """TASK 1b: Generate using 1NN only - UNCHANGED"""
        print(f"🔍 1NN-ONLY GENERATION:")
        print(f"Question: {question}")
        
        current_context = question  # Start with question as text (same as working version)
        generated_tokens = []
        
        for step in range(max_tokens):
            # Use working KNN retrieval - EXACT same as working version
            query_embedding = self.get_embedding_single(current_context)
            
            # FAISS search for 1NN (EXACT same as working version)
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 1
            )
            
            # EXACT SAME as working version: Just take 1NN target directly
            retrieved_target = self.datastore_targets[indices[0][0]]
            next_token_id = retrieved_target
            
            if step < 5:  # Show first few steps
                next_token_text = self.tokenizer.decode([next_token_id])
                distance = 1.0 - similarities[0][0]
                print(f"  Step {step}: '{next_token_text}' (1NN target: {next_token_id})")
                print(f"    1NN distance: {distance:.6f}")
            
            # Stop conditions
            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            # Add to generation
            generated_tokens.append(next_token_id)
            
            # 🔧 EXACT same spacing logic as working version
            question_tokens = self.tokenizer(question, add_special_tokens=False)['input_ids']
            full_tokens = question_tokens + generated_tokens
            current_context = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
            
            # Stop on sentence end
            if len(generated_tokens) > 10:
                recent_text = self.tokenizer.decode([next_token_id])
                if any(punct in recent_text for punct in ['.', '!', '?']):
                    break
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"✅ 1NN Answer: '{generated_text.strip()}'")
        return generated_text.strip()

    # 🔥 NEW TASK 1c: KNN-only generation with k neighbors
    def generate_knn_only(self, question: str, max_tokens: int = 30) -> str:
        """TASK 1c: Generate using KNN only (k neighbors with probability aggregation)"""
        print(f"🔍 KNN-ONLY GENERATION (k={self.k}):")
        print(f"Question: {question}")
        
        current_context = question  # Start with question as text
        generated_tokens = []
        
        for step in range(max_tokens):
            # Get k nearest neighbors
            query_embedding = self.get_embedding_single(current_context)
            
            # FAISS search for k neighbors
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), self.k
            )
            
            # Get target tokens and similarities for k neighbors
            target_tokens = [self.datastore_targets[i] for i in indices[0]]
            neighbor_similarities = similarities[0]
            
            # Convert similarities to probabilities using softmax
            neighbor_probs = F.softmax(torch.tensor(neighbor_similarities), dim=0).numpy()
            
            # Aggregate probabilities for tokens that appear multiple times
            vocab_size = len(self.tokenizer)
            token_probs = np.zeros(vocab_size)
            
            for token_id, prob in zip(target_tokens, neighbor_probs):
                if 0 <= token_id < vocab_size:
                    token_probs[token_id] += prob  # Sum probabilities for same token
            
            # Normalize to ensure it's a valid probability distribution
            if token_probs.sum() > 0:
                token_probs = token_probs / token_probs.sum()
            else:
                # Fallback: uniform distribution over retrieved tokens
                unique_tokens = list(set(target_tokens))
                for token_id in unique_tokens:
                    if 0 <= token_id < vocab_size:
                        token_probs[token_id] = 1.0 / len(unique_tokens)
            
            # Greedy selection: argmax
            next_token_id = np.argmax(token_probs)
            
            if step < 5:  # Show first few steps
                next_token_text = self.tokenizer.decode([next_token_id])
                distances = 1.0 - neighbor_similarities
                print(f"  Step {step}: '{next_token_text}' (KNN target: {next_token_id})")
                print(f"    Token prob: {token_probs[next_token_id]:.4f}")
                print(f"    Min distance: {np.min(distances):.6f}")
            
            # Stop conditions
            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            # Add to generation
            generated_tokens.append(next_token_id)
            
            # Same spacing logic as working version
            question_tokens = self.tokenizer(question, add_special_tokens=False)['input_ids']
            full_tokens = question_tokens + generated_tokens
            current_context = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
            
            # Stop on sentence end
            if len(generated_tokens) > 10:
                recent_text = self.tokenizer.decode([next_token_id])
                if any(punct in recent_text for punct in ['.', '!', '?']):
                    break
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"✅ KNN Answer: '{generated_text.strip()}'")
        return generated_text.strip()
    
    # TASK 2a: Combined LM + KNN generation - UNCHANGED
    def generate_combined(self, question: str, max_tokens: int = 30) -> str:
        """TASK 2a: Generate using Combined LM + KNN - UNCHANGED"""
        print(f"⚖️ COMBINED LM+KNN GENERATION (λ={self.lambda_weight}) - GREEDY:")
        print(f"Question: {question}")
        
        # Start with contexts as text (same as working version)
        lm_prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        knn_context = question  # Start with question as text
        generated_tokens = []
        
        for step in range(max_tokens):
            # Get LM distribution - build from generated tokens each time
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                current_lm_prompt = f"{lm_prompt} {generated_text}"
            else:
                current_lm_prompt = lm_prompt
                
            with torch.no_grad():
                encoding = self.tokenizer(current_lm_prompt, return_tensors='pt', truncation=True, max_length=1024)
                input_ids = encoding['input_ids'].to(self.device)
                
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
                lm_probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            # Get KNN distribution using EXACT same context building as working version
            query_embedding = self.get_embedding_single(knn_context)
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), self.k
            )
            
            target_tokens = [self.datastore_targets[i] for i in indices[0]]
            
            # Use proper softmax normalization on similarities
            neighbor_probs = F.softmax(torch.tensor(similarities[0]), dim=0).numpy()
            
            vocab_size = len(self.tokenizer)
            knn_probs = np.zeros(vocab_size)
            
            for token_id, prob in zip(target_tokens, neighbor_probs):
                if 0 <= token_id < vocab_size:
                    knn_probs[token_id] += prob
            
            if knn_probs.sum() > 0:
                knn_probs = knn_probs / knn_probs.sum()
            else:
                knn_probs = np.ones(vocab_size) / vocab_size
            
            # Combine distributions
            combined_probs = (1.0 - self.lambda_weight) * lm_probs + self.lambda_weight * knn_probs
            combined_probs = combined_probs / combined_probs.sum()
            
            # Use GREEDY DETERMINISTIC selection
            next_token_id = np.argmax(combined_probs)
            
            if step < 5:  # Show first few steps
                next_token_text = self.tokenizer.decode([next_token_id])
                lm_prob = lm_probs[next_token_id]
                knn_prob = knn_probs[next_token_id]
                combined_prob = combined_probs[next_token_id]
                print(f"  Step {step}: '{next_token_text}' | LM:{lm_prob:.4f} KNN:{knn_prob:.4f} Combined:{combined_prob:.4f} [GREEDY]")
            
            # Stop conditions
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            # Add to generation
            generated_tokens.append(next_token_id)
            
            # EXACT same spacing logic as working version for KNN context
            question_tokens = self.tokenizer(question, add_special_tokens=False)['input_ids']
            full_tokens = question_tokens + generated_tokens
            knn_context = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
            
            # Stop on sentence end
            if len(generated_tokens) > 10:
                recent_text = self.tokenizer.decode([next_token_id])
                if any(punct in recent_text for punct in ['.', '!', '?']):
                    break
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"✅ Combined LM+KNN Answer: '{generated_text.strip()}'")
        return generated_text.strip()

    # 🔥 NEW TASK 2b: Combined LM + 1NN generation
    def generate_combined_1nn(self, question: str, max_tokens: int = 75) -> str:
        """TASK 2b: Generate using Combined LM + 1NN"""
        print(f"⚖️ COMBINED LM+1NN GENERATION (λ={self.lambda_weight}) - GREEDY:")
        print(f"Question: {question}")
        
        # Start with contexts as text (same as working version)
        # lm_prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        lm_prompt = f"<s>[INST] Question: {question}\n\nGive a direct and concise answer to the question above. [/INST]"
        knn_context = question  # Start with question as text
        generated_tokens = []
        
        for step in range(max_tokens):
            # Get LM distribution - build from generated tokens each time
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                current_lm_prompt = f"{lm_prompt} {generated_text}"
            else:
                current_lm_prompt = lm_prompt
                
            with torch.no_grad():
                encoding = self.tokenizer(current_lm_prompt, return_tensors='pt', truncation=True, max_length=1024)
                input_ids = encoding['input_ids'].to(self.device)
                
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
                lm_probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            # Get 1NN distribution
            query_embedding = self.get_embedding_single(knn_context)
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 1  # Only 1 neighbor
            )
            
            # Get the 1NN target token
            target_token = self.datastore_targets[indices[0][0]]
            
            # Create 1NN distribution: prob=1 for target token, prob=0 for all others
            vocab_size = len(self.tokenizer)
            knn_probs = np.zeros(vocab_size)
            if 0 <= target_token < vocab_size:
                knn_probs[target_token] = 1.0  # Probability 1 for 1NN token
            
            # Combine distributions
            combined_probs = (1.0 - self.lambda_weight) * lm_probs + self.lambda_weight * knn_probs
            combined_probs = combined_probs / combined_probs.sum()
            
            # Use GREEDY DETERMINISTIC selection
            next_token_id = np.argmax(combined_probs)
            
            if step < 5:  # Show first few steps
                next_token_text = self.tokenizer.decode([next_token_id])
                lm_prob = lm_probs[next_token_id]
                knn_prob = knn_probs[next_token_id]
                combined_prob = combined_probs[next_token_id]
                distance = 1.0 - similarities[0][0]
                print(f"  Step {step}: '{next_token_text}' | LM:{lm_prob:.4f} 1NN:{knn_prob:.4f} Combined:{combined_prob:.4f} [GREEDY]")
                print(f"    1NN distance: {distance:.6f}")
            
            # Stop conditions
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            # Add to generation
            generated_tokens.append(next_token_id)
            
            # EXACT same spacing logic as working version for KNN context
            question_tokens = self.tokenizer(question, add_special_tokens=False)['input_ids']
            full_tokens = question_tokens + generated_tokens
            knn_context = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
            
            # Stop on sentence end
            # if len(generated_tokens) > 15:
            #     recent_text = self.tokenizer.decode([next_token_id])
            #     if any(punct in recent_text for punct in ['.', '!', '?']):
            #         break
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"✅ Combined LM+1NN Answer: '{generated_text.strip()}'")
        return generated_text.strip()
    
    def evaluate_on_questions(self, qa_pairs: List[dict]):
        """Evaluate all FIVE generation modes"""
        print("\n🧪 FIVE-WAY EVALUATION")
        print("=" * 80)
        sys.stdout.flush()
        
        results = []
        
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa['question']
            ground_truth = qa['answer']
            
            print(f"\n{'='*10} QUESTION {qa_idx + 1} {'='*10}")
            print(f"Q: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"{'='*50}")
            
            # TASK 1a: LM-only generation
            lm_answer = self.generate_lm_only(question)
            print()
            
            # TASK 1b: 1NN-only generation  
            one_nn_answer = self.generate_1nn_only(question)
            print()
            
            # 🔥 NEW TASK 1c: KNN-only generation
            knn_answer = self.generate_knn_only(question)
            print()
            
            # TASK 2a: Combined LM+KNN generation
            combined_answer = self.generate_combined(question)
            print()
            
            # 🔥 NEW TASK 2b: Combined LM+1NN generation
            combined_1nn_answer = self.generate_combined_1nn(question)
            
            # Summary
            print(f"\n📊 SUMMARY FOR QUESTION {qa_idx + 1}:")
            print(f"   Ground Truth:   '{ground_truth}'")
            print(f"   1a-LM:          '{lm_answer}'")
            print(f"   1b-1NN:         '{one_nn_answer}'")
            print(f"   1c-KNN:         '{knn_answer}'")
            print(f"   2a-Combined:    '{combined_answer}'")
            print(f"   2b-Combined1NN: '{combined_1nn_answer}'")
            print(f"{'='*80}")
            
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                # 'lm_only': lm_answer,
                # 'one_nn_only': one_nn_answer,
                # 'knn_only': knn_answer,
                # 'combined_lm_knn': combined_answer,
                'combined_lm_1nn': combined_1nn_answer
            })
        
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
    
    def run_complete_pipeline(self, train_file: str, test_file: str):
        """Run complete pipeline"""
        print(f"\n🎯 KNN-LM PIPELINE (FIVE TASK VERSION)")
        print(f"📁 Train: {train_file}")
        print(f"📁 Test: {test_file}")
        
        # Load data
        train_data = self.load_qa_data(train_file)
        test_data = self.load_qa_data(test_file)
        
        if not train_data or not test_data:
            print("❌ Missing data files")
            return
        
        # Build datastore (working version)
        self.build_datastore(train_data)
        
        # Test all FIVE generation modes
        results = self.evaluate_on_questions(test_data)
        
        print(f"\n🎯 PIPELINE COMPLETE!")
        print(f"   Five separate tasks completed successfully")
        return results
    
    def cleanup(self):
        """Cleanup"""
        for handle in self.hook_handles:
            handle.remove()
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lambda-weight', type=float, default=0.25)
    parser.add_argument('--batch-size', type=int, default=256)
    
    args = parser.parse_args()
    
    generator = None
    try:
        generator = FiveTaskKNNLMGenerator(
            batch_size=args.batch_size,
            k=args.k,
            lambda_weight=args.lambda_weight
        )
        
        results = generator.run_complete_pipeline(args.train_file, args.test_file)
        print("\n✅ ALL FIVE TASKS COMPLETED!")
                # Save results to specified path

        if results:
            import os
            # output_dir = "results/private/tofu/fixed_lambda_pretrained_embedding"
            output_dir = "results/public/fixed_lambda_pretrained_embedding"
            output_file = os.path.join(output_dir, "fixed_lambda_pretrained_embedding_lambda_0_75.json")
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to: {output_file}")
            print(f"📊 Saved {len(results)} question-answer pairs")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if generator:
            generator.cleanup()

if __name__ == "__main__":
    main()