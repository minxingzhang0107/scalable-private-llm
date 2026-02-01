#!/usr/bin/env python3
"""
KNN-LM Generation Time Analysis: Breakdown for different k values
Analyzes: Total time, FAISS search time, Distribution computation time
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

class KNNLMTimeAnalysisGenerator:
    """KNN-LM generator with detailed time analysis for different k values"""
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 adapter_path: str = "./model_checkpoints/mistral_tofu_lora_fixed_20250722_133126",
                 batch_size: int = 256, k: int = 10, upper_lambda: float = 1.0, 
                 lower_lambda: float = 0.0, distance_threshold: float = 0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.k = k
        self.upper_lambda = upper_lambda
        self.lower_lambda = lower_lambda
        self.distance_threshold = distance_threshold
        
        # Time tracking arrays
        self.total_times = []
        self.faiss_times = []
        self.dist_computation_times = []
        
        print("⚡ KNN-LM TIME ANALYSIS GENERATOR")
        print("=" * 70)
        print(f"Base Model: {model_path}")
        print(f"Fine-tuned Adapter: {adapter_path}")
        print(f"Device: {self.device}")
        print(f"K neighbors: {k}")
        print(f"Upper lambda (near): {upper_lambda}")
        print(f"Lower lambda (far): {lower_lambda}")
        print(f"Distance threshold: {distance_threshold}")
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
        
        print("🎯 Loading fine-tuned LoRA adapter for embeddings...")
        sys.stdout.flush()
        self.embedding_model = PeftModel.from_pretrained(base_embedding_model, adapter_path)
        self.embedding_model.eval()
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        
        # Setup hook for embedding extraction
        self.embeddings = []
        self.hook_handles = []
        self._setup_hook()
        
        # FAISS datastore
        self.faiss_index = None
        self.datastore_contexts = []
        self.datastore_targets = []
        
        print("✅ Initialization complete")
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
        """Setup embedding hook"""
        print("🔗 Setting up embedding hook...")
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
        print("✅ Hook registered")
        sys.stdout.flush()
    
    def get_embedding_single(self, text: str) -> np.ndarray:
        """Extract embedding for single text"""
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
        """Extract embeddings using SMART BATCHING"""
        print(f"⚡ SMART BATCH processing {len(contexts)} contexts")
        sys.stdout.flush()
        
        all_embeddings = []
        num_batches = (len(contexts) + self.batch_size - 1) // self.batch_size
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(contexts))
            batch_contexts = contexts[start_idx:end_idx]
            
            if batch_idx % 20 == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Batch {batch_idx + 1}/{num_batches} ({progress:.1f}%)")
                sys.stdout.flush()
            
            with torch.no_grad():
                self._set_deterministic()
                
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
                    batch_embeddings = []
                    
                    for i, seq_length in enumerate(batch_seq_lengths):
                        embedding = hook_output[i, seq_length]
                        embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
                        batch_embeddings.append(embedding.squeeze().cpu().numpy())
                    
                    all_embeddings.extend(batch_embeddings)
        
        total_time = time.time() - start_time
        print(f"✅ SMART BATCH complete! Time: {total_time:.1f}s")
        sys.stdout.flush()
        
        return np.array(all_embeddings)
    
    def create_context_target_pairs(self, question: str, answer: str) -> List[Tuple[str, int]]:
        """Create progressive context-target pairs"""
        
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
        """Build datastore - UNCHANGED"""
        print("\n🔨 BUILDING DATASTORE")
        print("-" * 70)
        sys.stdout.flush()
        
        start_time = time.time()
        
        all_contexts = []
        all_targets = []
        
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa['question']
            answer = qa['answer']
            
            if qa_idx % 500 == 0:
                progress_pct = (qa_idx + 1) / len(qa_pairs) * 100
                print(f"Processing Q&A {qa_idx + 1}/{len(qa_pairs)} ({progress_pct:.1f}%)")
                sys.stdout.flush()
            
            pairs = self.create_context_target_pairs(question, answer)
            
            for context, target in pairs:
                all_contexts.append(context)
                all_targets.append(target)
        
        pairs_time = time.time()
        print(f"✅ Created {len(all_contexts)} pairs in {pairs_time - start_time:.1f}s")
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
    
    def generate_with_time_analysis(self, question: str, max_tokens: int = 75) -> Tuple[str, Dict]:
        """Generate with detailed time tracking for each prefix"""
        print(f"\n⏱️ TIME ANALYSIS GENERATION (k={self.k}):")
        print(f"Question: {question}")
        
        lm_prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        knn_context = question
        generated_tokens = []
        
        # Reset time tracking
        self.total_times = []
        self.faiss_times = []
        self.dist_computation_times = []
        
        for step in range(max_tokens):
            # Start total time tracking
            step_start = time.time()
            
            # Get LM distribution
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                current_lm_prompt = f"{lm_prompt} {generated_text}"
            else:
                current_lm_prompt = lm_prompt
                
            with torch.no_grad():
                encoding = self.tokenizer(current_lm_prompt, return_tensors='pt', truncation=True, max_length=1024)
                input_ids = encoding['input_ids'].to(self.device)
                
                outputs = self.lm_model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
                lm_probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            # Get query embedding
            query_embedding = self.get_embedding_single(knn_context)
            
            # FAISS search - TIME THIS
            faiss_start = time.time()
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), self.k
            )
            faiss_time = time.time() - faiss_start
            
            # Distribution computation - TIME THIS
            dist_start = time.time()
            
            vocab_size = len(self.tokenizer)
            knn_probs = np.zeros(vocab_size)
            
            if self.k == 1:
                # Simple case: single neighbor
                target_token = self.datastore_targets[indices[0][0]]
                if 0 <= target_token < vocab_size:
                    knn_probs[target_token] = 1.0
            else:
                # Multiple neighbors: combine with distances
                for i in range(self.k):
                    target_token = self.datastore_targets[indices[0][i]]
                    similarity = similarities[0][i]
                    
                    if 0 <= target_token < vocab_size:
                        knn_probs[target_token] += similarity
                
                # Normalize
                if knn_probs.sum() > 0:
                    knn_probs = knn_probs / knn_probs.sum()
            
            # Dynamic lambda
            shortest_distance = 1.0 - similarities[0][0]
            if shortest_distance < self.distance_threshold:
                dynamic_lambda = self.upper_lambda
            else:
                dynamic_lambda = self.lower_lambda
            
            # Combine distributions
            combined_probs = (1.0 - dynamic_lambda) * lm_probs + dynamic_lambda * knn_probs
            combined_probs = combined_probs / combined_probs.sum()
            
            dist_computation_time = time.time() - dist_start
            
            # Total time for this step
            step_total_time = time.time() - step_start
            
            # Record times
            self.total_times.append(step_total_time)
            self.faiss_times.append(faiss_time)
            self.dist_computation_times.append(dist_computation_time)
            
            # Select next token
            next_token_id = np.argmax(combined_probs)
            
            if step < 3:
                print(f"  Step {step}: Total={step_total_time*1000:.2f}ms, FAISS={faiss_time*1000:.2f}ms, Dist={dist_computation_time*1000:.2f}ms")
            
            # Stop conditions
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
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
        
        # Compute statistics
        time_stats = {
            'k': self.k,
            'num_steps': len(self.total_times),
            'total_time_avg': np.mean(self.total_times),
            'total_time_std': np.std(self.total_times),
            'faiss_time_avg': np.mean(self.faiss_times),
            'faiss_time_std': np.std(self.faiss_times),
            'dist_computation_time_avg': np.mean(self.dist_computation_times),
            'dist_computation_time_std': np.std(self.dist_computation_times)
        }
        
        print(f"\n📊 Time Statistics (k={self.k}):")
        print(f"  Total steps: {time_stats['num_steps']}")
        print(f"  Total time per step: {time_stats['total_time_avg']*1000:.2f} ± {time_stats['total_time_std']*1000:.2f} ms")
        print(f"  FAISS search time: {time_stats['faiss_time_avg']*1000:.2f} ± {time_stats['faiss_time_std']*1000:.2f} ms")
        print(f"  Dist computation time: {time_stats['dist_computation_time_avg']*1000:.2f} ± {time_stats['dist_computation_time_std']*1000:.2f} ms")
        
        return generated_text.strip(), time_stats
    
    def evaluate_time_analysis(self, qa_pairs: List[dict], output_file: str):
        """Evaluate with time analysis"""
        print("\n🧪 TIME ANALYSIS EVALUATION")
        print("=" * 80)
        sys.stdout.flush()
        
        results = []
        all_stats = []
        
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa['question']
            ground_truth = qa['answer']
            
            print(f"\n{'='*10} QUESTION {qa_idx + 1}/{len(qa_pairs)} {'='*10}")
            print(f"Q: {question}")
            
            generated_answer, time_stats = self.generate_with_time_analysis(question)
            
            result_entry = qa.copy()
            result_entry['answer'] = generated_answer
            result_entry['time_stats'] = time_stats
            results.append(result_entry)
            all_stats.append(time_stats)
        
        # Aggregate statistics across all questions
        aggregate_stats = {
            'k': self.k,
            'num_questions': len(all_stats),
            'total_steps': sum(s['num_steps'] for s in all_stats),
            'avg_steps_per_question': np.mean([s['num_steps'] for s in all_stats]),
            'overall_total_time_avg': np.mean([s['total_time_avg'] for s in all_stats]),
            'overall_total_time_std': np.std([s['total_time_avg'] for s in all_stats]),
            'overall_faiss_time_avg': np.mean([s['faiss_time_avg'] for s in all_stats]),
            'overall_faiss_time_std': np.std([s['faiss_time_avg'] for s in all_stats]),
            'overall_dist_computation_time_avg': np.mean([s['dist_computation_time_avg'] for s in all_stats]),
            'overall_dist_computation_time_std': np.std([s['dist_computation_time_avg'] for s in all_stats])
        }
        
        print(f"\n{'='*80}")
        print(f"📊 AGGREGATE TIME STATISTICS (k={self.k}):")
        print(f"  Total questions: {aggregate_stats['num_questions']}")
        print(f"  Total steps: {aggregate_stats['total_steps']}")
        print(f"  Avg steps per question: {aggregate_stats['avg_steps_per_question']:.1f}")
        print(f"  Overall total time per step: {aggregate_stats['overall_total_time_avg']*1000:.2f} ± {aggregate_stats['overall_total_time_std']*1000:.2f} ms")
        print(f"  Overall FAISS search time: {aggregate_stats['overall_faiss_time_avg']*1000:.2f} ± {aggregate_stats['overall_faiss_time_std']*1000:.2f} ms")
        print(f"  Overall dist computation time: {aggregate_stats['overall_dist_computation_time_avg']*1000:.2f} ± {aggregate_stats['overall_dist_computation_time_std']*1000:.2f} ms")
        
        # Save results
        output_data = {
            'results': results,
            'aggregate_stats': aggregate_stats
        }
        
        print(f"\n💾 Saving results to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved!")
        
        return results, aggregate_stats
    
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
    
    def run_time_analysis_pipeline(self, train_file: str, test_file: str, output_file: str):
        """Run time analysis pipeline"""
        print(f"\n🎯 TIME ANALYSIS PIPELINE (k={self.k})")
        print(f"📁 Train: {train_file}")
        print(f"📁 Test: {test_file}")
        print(f"💾 Output: {output_file}")
        
        train_data = self.load_qa_data(train_file)
        test_data = self.load_qa_data(test_file)
        
        if not train_data or not test_data:
            print("❌ Missing data files")
            return
        
        # Build datastore (unchanged)
        self.build_datastore(train_data)
        
        # Run time analysis
        results, aggregate_stats = self.evaluate_time_analysis(test_data, output_file)
        
        print(f"\n🎯 TIME ANALYSIS COMPLETE!")
        print(f"   k = {self.k}")
        print(f"   Results saved to: {output_file}")
        return results, aggregate_stats
    
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
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--adapter-path', default='./model_checkpoints/mistral_tofu_lora_fixed_20250722_133126')
    parser.add_argument('--k', type=int, required=True, help='Number of neighbors for analysis')
    parser.add_argument('--upper-lambda', type=float, default=1.0)
    parser.add_argument('--lower-lambda', type=float, default=0.0)
    parser.add_argument('--distance-threshold', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=256)
    
    args = parser.parse_args()
    
    generator = None
    try:
        generator = KNNLMTimeAnalysisGenerator(
            adapter_path=args.adapter_path,
            batch_size=args.batch_size,
            k=args.k,
            upper_lambda=args.upper_lambda,
            lower_lambda=args.lower_lambda,
            distance_threshold=args.distance_threshold
        )
        
        results, stats = generator.run_time_analysis_pipeline(
            args.train_file, args.test_file, args.output_file
        )
        print(f"\n✅ TIME ANALYSIS COMPLETED FOR k={args.k}!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if generator:
            generator.cleanup()

if __name__ == "__main__":
    main()