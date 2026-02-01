#!/usr/bin/env python3
"""
KNN-LM Generation with FINE-TUNED EMBEDDINGS: DISTANCE HISTOGRAM ANALYSIS
Same as original code but collects shortest distances and plots histograms instead of saving JSON
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
from typing import List, Tuple, Dict
import sys
import faiss
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

class FiveTaskDynamicLambdaKNNLMGenerator:
    """KNN-LM generator with FINE-TUNED EMBEDDINGS and DYNAMIC LAMBDA per token - Distance Analysis Version"""
    
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
        
        # Lists to store shortest distances for histogram analysis
        self.private_distances = []
        self.public_distances = []
        
        print("⚡ KNN-LM GENERATOR (FINE-TUNED EMBEDDINGS + DYNAMIC LAMBDA) - DISTANCE ANALYSIS")
        print("=" * 70)
        print(f"Base Model: {model_path}")
        print(f"Fine-tuned Adapter: {adapter_path}")
        print(f"Device: {self.device}")
        print(f"K neighbors: {k}")
        print(f"Upper lambda (near): {upper_lambda}")
        print(f"Lower lambda (far): {lower_lambda}")
        print(f"Distance threshold: {distance_threshold}")
        print(f"🔥 LM: Pre-trained | Embeddings: Fine-tuned")
        print(f"TASK: Distance Histogram Analysis")
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
    
    def generate_combined_1nn_dynamic_collect_distances(self, question: str, max_tokens: int = 75, dataset_type: str = "unknown") -> Tuple[str, List[float]]:
        """TASK 2b: Generate using Combined LM + 1NN with DYNAMIC LAMBDA per token - COLLECT DISTANCES"""
        print(f"⚖️ COMBINED LM+1NN GENERATION (DYNAMIC λ) - GREEDY - {dataset_type.upper()}:")
        print(f"Question: {question}")
        print(f"Upper λ: {self.upper_lambda} (distance < {self.distance_threshold})")
        print(f"Lower λ: {self.lower_lambda} (distance >= {self.distance_threshold})")
        
        # Start with contexts as text
        # lm_prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        # lm_prompt = f"<s>[INST] Question: {question}\n\nGive a direct and concise answer to the question above. [/INST]"
        if dataset_type.lower() == "private":
            lm_prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        else:  # public dataset
            lm_prompt = f"<s>[INST] Question: {question}\n\nGive a direct and concise answer to the question above. [/INST]"
        knn_context = question  # Start with question as text
        generated_tokens = []
        distances = []  # Collect distances for this question
        
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
            distances.append(float(shortest_distance))  # COLLECT DISTANCE
            
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
            if dataset_type.lower() == "private":
                if len(generated_tokens) > 15:
                    recent_text = self.tokenizer.decode([next_token_id])
                    if any(punct in recent_text for punct in ['.', '!', '?']):
                        break
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"✅ Combined LM+1NN Answer: '{generated_text.strip()}'")
        print(f"📊 Collected {len(distances)} distances for this question")
        return generated_text.strip(), distances
    
    def evaluate_on_questions_collect_distances(self, qa_pairs: List[dict], dataset_type: str = "unknown"):
        """Evaluate ONLY Combined LM+1NN generation and COLLECT DISTANCES"""
        print(f"\n🧪 COMBINED LM+1NN EVALUATION - DISTANCE COLLECTION ({dataset_type.upper()})")
        print("=" * 80)
        sys.stdout.flush()
        
        all_distances = []
        
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa['question']
            ground_truth = qa['answer']
            
            print(f"\n{'='*10} QUESTION {qa_idx + 1} {'='*10}")
            print(f"Q: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"{'='*50}")
            
            # Combined LM+1NN generation with DYNAMIC LAMBDA - COLLECT DISTANCES
            combined_1nn_answer, question_distances = self.generate_combined_1nn_dynamic_collect_distances(question, dataset_type=dataset_type)
            
            # Add distances to the appropriate list
            all_distances.extend(question_distances)
            
            print(f"{'='*80}")
        
        print(f"\n📊 Total distances collected for {dataset_type}: {len(all_distances)}")
        return all_distances
    
    # def plot_distance_histogram(self, output_path: str = "distance_histogram.png"):
    #     """Plot histogram of shortest distances"""
        
    #     # Set up the plot with publication-quality styling
    #     plt.style.use('seaborn-v0_8-paper')  # Clean, professional style
    #     plt.figure(figsize=(10, 7))
        
    #     # Plot histograms with professional colors
    #     bins = np.linspace(0, 1, 50)
        
    #     if self.private_distances:
    #         plt.hist(self.private_distances, bins=bins, alpha=0.6, color='#d62728', 
    #                 label='Private Data', density=True, edgecolor='black', linewidth=0.5)
        
    #     if self.public_distances:
    #         plt.hist(self.public_distances, bins=bins, alpha=0.6, color='#1f77b4', 
    #                 label='Public Data', density=True, edgecolor='black', linewidth=0.5)
        
    #     # Add vertical line for threshold
    #     plt.axvline(x=self.distance_threshold, color='black', linestyle='--', 
    #             linewidth=2.5, label=f'Threshold = {self.distance_threshold}')
        
    #     # Customize plot with large, readable fonts
    #     plt.xlabel('Shortest Distance to Nearest Neighbor', fontsize=28, fontweight='bold')
    #     plt.ylabel('Density', fontsize=28, fontweight='bold')
    #     # plt.title('Distribution of Shortest Distances\nDuring KNN-LM Generation', 
    #     #         fontsize=22, fontweight='bold', pad=20)
        
    #     # Increase tick label sizes
    #     plt.xticks(fontsize=25)
    #     plt.yticks(fontsize=25)
        
    #     # Enhance legend
    #     plt.legend(fontsize=30, frameon=True, shadow=True, fancybox=True, 
    #             loc='upper right', framealpha=0.95)
        
    #     # Add subtle grid
    #     plt.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        
    #     # Remove statistics text box for cleaner look
        
    #     plt.tight_layout()
        
    #     # Save plot in high resolution
    #     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    #     # Also save as PDF (vector format - best for papers!)
    #     pdf_path = output_path.replace('.png', '.pdf')
    #     plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        
    #     plt.show()
    def plot_distance_histogram(self, output_path: str = "distance_histogram.png"):
        """Plot histogram of shortest distances"""
        
        # Set publication-quality defaults matching first figure
        plt.rcParams.update({
            'font.size': 16,
            'font.family': 'sans-serif',
            'axes.labelsize': 20,
            'axes.titlesize': 22,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 15,
            'figure.figsize': (9, 5.5),
            'lines.linewidth': 2.5,
            'lines.markersize': 7,
            'grid.alpha': 0.3,
            'axes.grid': True,
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(9, 5.7))
        
        # Plot histograms with colors from first figure
        bins = np.linspace(0, 1, 50)
        
        if self.private_distances:
            ax.hist(self.private_distances, bins=bins, alpha=0.2, color='#D62246', 
                    label='Private Data', density=True, edgecolor='black', linewidth=0.5)
        
        if self.public_distances:
            ax.hist(self.public_distances, bins=bins, alpha=0.2, color='#06A77D', 
                    label='Public Data', density=True, edgecolor='black', linewidth=0.5)
        
        # Add vertical line for threshold
        ax.axvline(x=self.distance_threshold, color='black', linestyle='--', 
                linewidth=2.5, label=f'Threshold = {self.distance_threshold}')
        
        # Customize plot to match first figure style
        ax.set_xlabel('Shortest Distance to Nearest Neighbor', fontsize=22, fontweight='bold')
        ax.set_ylabel('Density', fontsize=22, fontweight='bold')
        ax.set_title('Shortest Distance Distribution', 
                    fontsize=24, fontweight='bold', pad=15)
        
        # Increase tick label sizes to match
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # Legend matching first figure style
        ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fontsize=22)
        
        # Grid matching first figure
        ax.grid(True, linestyle='--', alpha=0.3, axis='both')
        
        plt.tight_layout()
        
        # Save plot in high resolution
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Also save as PDF
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        
        print(f"✅ Histogram saved to: {output_path}")
        print(f"✅ PDF version saved to: {pdf_path}")
        
        plt.close()
    
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
    
    def run_complete_pipeline(self, train_file: str, private_test_file: str, public_test_file: str, output_plot: str):
        """Run complete pipeline - DISTANCE COLLECTION VERSION"""
        print(f"\n🎯 KNN-LM DISTANCE ANALYSIS PIPELINE (COMBINED LM+1NN WITH FINE-TUNED EMBEDDINGS)")
        print(f"📁 Train: {train_file}")
        print(f"📁 Private Test: {private_test_file}")
        print(f"📁 Public Test: {public_test_file}")
        print(f"📊 Output Plot: {output_plot}")
        
        # Load data
        train_data = self.load_qa_data(train_file)
        private_data = self.load_qa_data(private_test_file) if private_test_file else []
        public_data = self.load_qa_data(public_test_file) if public_test_file else []
        
        if not train_data:
            print("❌ Missing train data files")
            return
        
        # Build datastore using fine-tuned embeddings
        self.build_datastore(train_data)
        
        # Test on private dataset and collect distances
        if private_data:
            print(f"\n🔴 PROCESSING PRIVATE DATASET ({len(private_data)} questions)")
            self.private_distances = self.evaluate_on_questions_collect_distances(private_data, "private")
        
        # Test on public dataset and collect distances
        if public_data:
            print(f"\n🔵 PROCESSING PUBLIC DATASET ({len(public_data)} questions)")
            self.public_distances = self.evaluate_on_questions_collect_distances(public_data, "public")
        
        # Plot histogram
        if self.private_distances or self.public_distances:
            self.plot_distance_histogram(output_plot)
        else:
            print("❌ No distances collected!")
        
        print(f"\n🎯 DISTANCE ANALYSIS PIPELINE COMPLETE!")
        print(f"   Combined LM+1NN completed with FINE-TUNED EMBEDDINGS + DYNAMIC LAMBDA!")
        print(f"   🤖 LM: Pre-trained | 🎯 Embeddings: Fine-tuned")
        print(f"   📊 Histogram saved to: {output_plot}")
        return self.private_distances, self.public_distances
    
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
    parser.add_argument('--private-test-file', required=True)
    parser.add_argument('--public-test-file', required=True)
    parser.add_argument('--output-plot', default='results/analysis/distance_histogram.png', 
                       help='Output path for histogram plot')
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
    
    generator = None
    try:
        generator = FiveTaskDynamicLambdaKNNLMGenerator(
            adapter_path=args.adapter_path,
            batch_size=args.batch_size,
            k=args.k,
            upper_lambda=args.upper_lambda,
            lower_lambda=args.lower_lambda,
            distance_threshold=args.distance_threshold
        )
        
        results = generator.run_complete_pipeline(args.train_file, args.private_test_file, args.public_test_file, args.output_plot)
        print("\n✅ DISTANCE HISTOGRAM ANALYSIS COMPLETED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if generator:
            generator.cleanup()

if __name__ == "__main__":
    main()