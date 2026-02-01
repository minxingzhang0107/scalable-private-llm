#!/usr/bin/env python3
"""
KNN-LM Generation: FINAL-FIXED-VERSION
1. Re-implemented "smart batching" with MANUAL RIGHT-PADDING,
   mirroring the user's working small-scale code.
2. This ensures datastore keys (batched, right-padded) and
   queries (single, no-pad) are consistent.
3. LM pass still uses KV cache for speed.
4. All typos fixed.
5. Added user-requested stop-word condition.
6. Added performance timing (avg, std) and progress tracking for evaluation.
7. REMOVED step-by-step similarity printing for clean logs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from typing import List, Tuple, Dict, Generator
import sys
import faiss
import argparse
import os

class StreamingKNNLMGenerator:
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 batch_size: int = 768, k: int = 1, upper_lambda: float = 1.0, 
                 lower_lambda: float = 0.0, distance_threshold: float = 0.4,
                 datastore_dir: str = "datastore",
                 use_ivf: bool = True,
                 num_gpus: int = 1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size # This is now for GPU batching
        self.k = k
        self.upper_lambda = upper_lambda
        self.lower_lambda = lower_lambda
        self.distance_threshold = distance_threshold
        self.datastore_dir = datastore_dir
        self.use_ivf = use_ivf
        self.num_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        
        print("⚡ FINAL-FIXED KNN-LM (Smart Batching with Manual Right-Padding + Perf.)")
        print("=" * 70)
        print(f"GPUs: {self.num_gpus}, Batch: {self.batch_size}, k: {k}, IVF: {use_ivf}")
        sys.stdout.flush()
        
        self._set_deterministic()
        
        print("Loading model...")
        sys.stdout.flush()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Set padding side to right for consistency, although we pad manually
        self.tokenizer.padding_side = "right" 
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if self.num_gpus == 1 else "balanced",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.faiss_index = None
        self.datastore_targets = None
        self.embedding_dim = self.model.config.hidden_size
        
        print(f"Embedding Dim: {self.embedding_dim}")
        print("✅ Ready")
        sys.stdout.flush()
    
    def _set_deterministic(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # --- START OF REFACTOR ---
    
    def _get_embedding_single(self, context: str) -> np.ndarray:
        """
        Gets a single embedding for a query (no padding).
        This is used by the generator.
        """
        with torch.no_grad():
            self._set_deterministic()
            
            encoding = self.tokenizer(
                [context], # List of 1
                add_special_tokens=False,
                truncation=True, 
                max_length=512, 
                padding=False, # NO PADDING
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.model.device)
            attention_mask = encoding['attention_mask'].to(self.model.device)
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True,
                    use_cache=False
                )
            
            if outputs.hidden_states:
                last_hidden_state = outputs.hidden_states[-1]
                normed_hidden_state = self.model.model.norm(last_hidden_state) # Apply final norm
                embedding = normed_hidden_state[0, -1] # Get last token
                embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
                return embedding.squeeze().cpu().numpy()
            else:
                raise RuntimeError("No hidden states")

    def _get_embeddings_batched(self, contexts: List[str]) -> np.ndarray:
        """
        Gets embeddings for a batch of datastore contexts.
        Uses MANUAL RIGHT-PADDING to match the user's working code.
        """
        with torch.no_grad():
            self._set_deterministic()
            
            batch_input_ids = []
            batch_attention_masks = []
            batch_seq_lengths = [] # Original lengths

            # 1. Tokenize each context individually (on CPU)
            for context in contexts:
                encoding = self.tokenizer(
                    context,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].squeeze(0) # [seq_len]
                attention_mask = encoding['attention_mask'].squeeze(0) # [seq_len]
                
                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
                batch_seq_lengths.append(attention_mask.sum().item() - 1)

            # 2. Find max_len and manually pad to the RIGHT (all on CPU)
            max_len = max(len(ids) for ids in batch_input_ids)
            
            padded_input_ids = []
            padded_attention_masks = []
            
            for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks):
                pad_length = max_len - len(input_ids)
                
                if pad_length > 0:
                    padded_ids = torch.cat([
                        input_ids,
                        torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
                    ])
                    padded_mask = torch.cat([
                        attention_mask,
                        torch.zeros(pad_length, dtype=attention_mask.dtype, device=attention_mask.device)
                    ])
                else:
                    padded_ids = input_ids
                    padded_mask = attention_mask
                
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            
            # 3. Stack (on CPU) and then move the *entire batch* to the GPU
            batch_input_ids_tensor = torch.stack(padded_input_ids).to(self.model.device)
            batch_attention_masks_tensor = torch.stack(padded_attention_masks).to(self.model.device)

            with torch.amp.autocast('cuda'):
                outputs = self.model(
                    input_ids=batch_input_ids_tensor, 
                    attention_mask=batch_attention_masks_tensor, 
                    output_hidden_states=True,
                    use_cache=False
                )
            
            # 4. Extract embeddings using *original* sequence lengths
            batch_embeddings = []
            if outputs.hidden_states:
                last_hidden_state = outputs.hidden_states[-1]
                normed_hidden_state = self.model.model.norm(last_hidden_state).cpu() # Apply final norm
                
                for i, seq_length_idx in enumerate(batch_seq_lengths):
                    embedding = normed_hidden_state[i, seq_length_idx] 
                    embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
                    batch_embeddings.append(embedding.squeeze().cpu().numpy())
            else:
                batch_embeddings = [np.zeros(self.embedding_dim) for _ in range(len(contexts))]
            
            return np.array(batch_embeddings)

    # --- END OF REFACTOR ---

    def qa_pair_generator(self, qa_pairs: List[dict]) -> Generator[Tuple[str, int], None, None]:
        # (This function is correct and unchanged)
        for qa_idx, qa in enumerate(qa_pairs):
            if qa_idx % 10000 == 0:
                print(f"  Processing Q&A {qa_idx + 1}/{len(qa_pairs)} ({100*(qa_idx+1)/len(qa_pairs):.1f}%)")
                sys.stdout.flush()
            
            q_encoding = self.tokenizer(qa['question'], add_special_tokens=False, return_tensors='pt')
            a_encoding = self.tokenizer(qa['answer'], add_special_tokens=False, return_tensors='pt')
            
            q_ids = q_encoding['input_ids'].squeeze().tolist()
            a_ids = a_encoding['input_ids'].squeeze().tolist()
            
            if not isinstance(q_ids, list):
                q_ids = [q_ids]
            if not isinstance(a_ids, list):
                a_ids = [a_ids]
            
            full_ids = q_ids + a_ids
            
            for i in range(len(q_ids), len(full_ids)):
                context_ids = full_ids[:i]
                target_id = full_ids[i]
                context_text = self.tokenizer.decode(context_ids, skip_special_tokens=True)
                yield (context_text, target_id)
    
    def estimate_total_pairs(self, qa_pairs: List[dict], sample_size: int = 100) -> int:
        # (This function is correct and unchanged)
        sample_size = min(sample_size, len(qa_pairs))
        total = 0
        for i, qa in enumerate(qa_pairs[:sample_size]):
            q_len = len(self.tokenizer(qa['question'], add_special_tokens=False)['input_ids'])
            a_len = len(self.tokenizer(qa['answer'], add_special_tokens=False)['input_ids'])
            total += a_len
        
        avg_pairs_per_qa = total / sample_size
        estimated_total = int(avg_pairs_per_qa * len(qa_pairs))
        return estimated_total
    
    def build_datastore_streaming(self, qa_pairs: List[dict]):
        print("\n🔨 BUILDING DATASTORE (TRUE STREAMING - Smart Batching)")
        print("-" * 70)
        sys.stdout.flush()
        
        start_time = time.time()
        
        print("Estimating total pairs...")
        estimated_total = self.estimate_total_pairs(qa_pairs)
        print(f"Estimated: ~{estimated_total:,} pairs")
        sys.stdout.flush()
        
        if self.use_ivf:
            print(f"Initializing IVF FAISS...")
            nlist = min(int(np.sqrt(estimated_total)), 4096)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            cpu_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            cpu_index.nprobe = 32
            needs_training = True
        else:
            print(f"Initializing Flat FAISS...")
            cpu_index = faiss.IndexFlatIP(self.embedding_dim)
            needs_training = False
        
        print("Keeping FAISS on CPU...")
        sys.stdout.flush()
        self.faiss_index = cpu_index
        print("✅ FAISS on CPU")
        sys.stdout.flush()
        
        datastore_targets_list = []
        context_batch_buffer = []
        
        pair_gen = self.qa_pair_generator(qa_pairs)
        build_start = time.time()
        total_processed = 0
        
        all_embeddings_list = []

        print("\n🚀 Streaming contexts and processing embeddings in batches...")
        sys.stdout.flush()
        
        for context, target in pair_gen:
            context_batch_buffer.append(context)
            datastore_targets_list.append(target)
            total_processed += 1
            
            if len(context_batch_buffer) == self.batch_size:
                embeddings = self._get_embeddings_batched(context_batch_buffer)
                all_embeddings_list.append(embeddings.astype(np.float32))
                context_batch_buffer = [] 
                
                if total_processed % (self.batch_size * 100) == 0:
                    elapsed = time.time() - build_start
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    progress = 100 * total_processed / estimated_total if estimated_total > 0 else 0
                    eta = (estimated_total - total_processed) / rate if rate > 0 else 0
                    print(f"  Processed {total_processed:,}/{estimated_total:,} ({progress:.1f}%) | {rate:.0f} pairs/s | ETA: {eta/60:.0f}m")
                    sys.stdout.flush()
                del embeddings
        
        if context_batch_buffer:
            embeddings = self._get_embeddings_batched(context_batch_buffer)
            all_embeddings_list.append(embeddings.astype(np.float32))
            context_batch_buffer = []
            del embeddings

        print(f"\n✅ Collected {total_processed} embeddings.")
        
        all_embeddings_np = np.vstack(all_embeddings_list)
        self.datastore_targets = np.array(datastore_targets_list, dtype=np.int32)
        del all_embeddings_list, datastore_targets_list

        if needs_training:
            IVF_TRAIN_SIZE = min(1000000, all_embeddings_np.shape[0])
            print(f"\nTraining IVF with {IVF_TRAIN_SIZE:,} vectors...")
            sys.stdout.flush()
            
            train_data = all_embeddings_np
            if all_embeddings_np.shape[0] > IVF_TRAIN_SIZE:
                indices = np.random.permutation(all_embeddings_np.shape[0])[:IVF_TRAIN_SIZE]
                train_data = all_embeddings_np[indices]

            self.faiss_index.train(train_data)
            print("✅ Training complete")
            sys.stdout.flush()
            del train_data

        print("\nAdding all embeddings to FAISS index...")
        self.faiss_index.add(all_embeddings_np)
        del all_embeddings_np
        
        print(f"\n✅ FAISS build complete with {self.faiss_index.ntotal:,} vectors")
        
        print("Saving datastore to disk...")
        os.makedirs(self.datastore_dir, exist_ok=True)
        
        faiss_index_path = os.path.join(self.datastore_dir, "datastore.index")
        targets_path = os.path.join(self.datastore_dir, "datastore_targets.npy")
        
        faiss.write_index(self.faiss_index, faiss_index_path)
        np.save(targets_path, self.datastore_targets)
        
        total_time = time.time() - start_time
        print(f"\n✅ BUILD COMPLETE & SAVED")
        print(f"   Pairs: {len(self.datastore_targets):,}")
        print(f"   FAISS: {self.faiss_index.ntotal:,}")
        print(f"   Time: {total_time/60:.1f} min")
        print(f"   Rate: {len(self.datastore_targets)/total_time:.0f} pairs/s")
        print(f"   Saved to: {self.datastore_dir}/")
        sys.stdout.flush()
    
    def load_datastore(self) -> bool:
        # (This function is correct and unchanged)
        faiss_index_path = os.path.join(self.datastore_dir, "datastore.index")
        targets_path = os.path.join(self.datastore_dir, "datastore_targets.npy")
        
        if os.path.exists(faiss_index_path) and os.path.exists(targets_path):
            print(f"\n✅ Loading existing datastore from {self.datastore_dir}/")
            sys.stdout.flush()
            
            load_start = time.time()
            self.faiss_index = faiss.read_index(faiss_index_path)
            self.datastore_targets = np.load(targets_path)
            load_time = time.time() - load_start
            
            print(f"   FAISS vectors: {self.faiss_index.ntotal:,}")
            print(f"   Target tokens: {len(self.datastore_targets):,}")
            print(f"   Load time: {load_time:.1f}s")
            sys.stdout.flush()
            return True
        
        return False
    
    def generate_combined_1nn_dynamic(self, question: str, max_tokens: int = 20) -> str:
        """
        FIXED: 
        1. LM pass uses KV cache for speed.
        2. k-NN pass uses the canonical, SINGLE embedding function.
        """
        
        # 1. Prepare LM (Instructed) Inputs
        lm_prompt = f"<s>[INST] Question: {question}\n\nGive a direct and concise answer to the question above. [/INST]"
        lm_encoding = self.tokenizer(lm_prompt, return_tensors='pt', truncation=True, max_length=1024)
        lm_input_ids = lm_encoding['input_ids'].to(self.device)
        lm_past_key_values = None # Cache for the LM

        # 2. Prepare k-NN (Raw) Inputs
        q_encoding = self.tokenizer(question, add_special_tokens=False, return_tensors='pt')
        q_ids = q_encoding['input_ids'].squeeze().tolist()
        if not isinstance(q_ids, list):
            q_ids = [q_ids]
            
        knn_context_string = self.tokenizer.decode(q_ids, skip_special_tokens=True)
        
        generated_tokens = []
        
        for step in range(max_tokens):
            with torch.no_grad():
                
                # --- LM Pass (FAST: Uses KV Cache) ---
                lm_outputs = self.model(
                    input_ids=lm_input_ids, 
                    past_key_values=lm_past_key_values, 
                    use_cache=True,
                    output_hidden_states=False
                )
                logits = lm_outputs.logits[0, -1, :]
                lm_past_key_values = lm_outputs.past_key_values # Update LM cache
                lm_probs = F.softmax(logits, dim=-1).cpu().numpy()
                
                # --- k-NN Pass (CORRECT: Uses canonical single func) ---
                query_embedding_np = self._get_embedding_single(knn_context_string)
            
            # --- k-NN Search (using the correct raw embedding) ---
            similarities, indices = self.faiss_index.search(
                query_embedding_np.reshape(1, -1).astype(np.float32), 
                self.k
            )
            
            # --- THIS IS THE FIX: REMOVED SIMILARITY PRINTING ---
            # if step < 3 or (1.0 - similarities[0][0]) < 0.001:
            #     print(f"  Step {step}: Similarity = {similarities[0][0]:.4f}")
            # --- END OF FIX ---

            # --- Distribution Combination ---
            shortest_distance = 1.0 - similarities[0][0]
            dynamic_lambda = self.upper_lambda if shortest_distance < self.distance_threshold else self.lower_lambda
            
            vocab_size = len(self.tokenizer)
            knn_probs = np.zeros(vocab_size)
            
            if self.k == 1:
                embedding_idx = int(indices[0][0])
                target_token = int(self.datastore_targets[embedding_idx])
                if 0 <= target_token < vocab_size:
                    knn_probs[target_token] = 1.0
            else:
                for i in range(self.k):
                    embedding_idx = int(indices[0][i])
                    target_token = int(self.datastore_targets[embedding_idx])
                    if 0 <= target_token < vocab_size:
                        knn_probs[target_token] += similarities[0][i]
                if knn_probs.sum() > 0:
                    knn_probs = knn_probs / knn_probs.sum()
            
            combined_probs = (1.0 - dynamic_lambda) * lm_probs + dynamic_lambda * knn_probs
            combined_probs = combined_probs / combined_probs.sum()
            next_token_id = np.argmax(combined_probs)
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token_id)
            
            # Stop on sentence end
            if len(generated_tokens) > 10:
                recent_text = self.tokenizer.decode([next_token_id])
                if any(punct in recent_text for punct in ['.', '!', '?']):
                    break

            # --- Update states for next iteration ---
            
            # 1. Update LM input (fast, just the new token)
            next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            lm_input_ids = next_token_tensor
            
            # 2. Update k-NN context string (must match datastore logic)
            full_raw_ids = q_ids + generated_tokens
            knn_context_string = self.tokenizer.decode(full_raw_ids, skip_special_tokens=True)
        
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    def evaluate_on_questions(self, qa_pairs: List[dict], output_file: str):
        print("\n🧪 EVALUATION")
        print("=" * 80)
        sys.stdout.flush()
        
        results = []
        generation_times = [] # To store duration of each generation
        
        for qa_idx, qa in enumerate(qa_pairs):
            # 1. Progress Tracking
            print(f"\n--- Processing Query {qa_idx + 1}/{len(qa_pairs)} ---")
            print(f"Q: {qa['question']}")
            
            # 2. Timing
            start_gen_time = time.time()
            answer = self.generate_combined_1nn_dynamic(qa['question'])
            end_gen_time = time.time()
            
            duration = end_gen_time - start_gen_time
            generation_times.append(duration)
            
            print(f"A: {answer}  (Time: {duration:.2f}s)")
            
            result_entry = qa.copy()
            result_entry['answer'] = answer
            results.append(result_entry)
        
        # Add timing statistics
        if generation_times:
            avg_time = np.mean(generation_times)
            std_time = np.std(generation_times)
            
            print("\n" + "=" * 80)
            print("📈 Evaluation Timing Summary:")
            print(f"  Total Queries:  {len(generation_times)}")
            print(f"  Average Time:   {avg_time:.4f}s per query")
            print(f"  Std. Deviation: {std_time:.4f}s")
            print("=" * 80)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved to {output_file}")
        return results

    def load_qa_data(self, filepath: str):
        # (This function is correct and unchanged)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} Q&A pairs")
        return data
    
    def run_complete_pipeline(self, train_file: str, test_file: str, output_file: str, force_rebuild: bool = False):
        # (This function is correct and unchanged)
        test_data = self.load_qa_data(test_file)
        
        if not test_data:
            print("❌ Missing test data")
            return
        
        datastore_loaded = False
        if not force_rebuild:
            datastore_loaded = self.load_datastore()
        
        if not datastore_loaded:
            print(f"\n❌ No datastore found or rebuild forced. Building from {train_file}...")
            sys.stdout.flush()
            train_data = self.load_qa_data(train_file)
            if not train_data:
                print("❌ Missing training data to build datastore")
                return
            self.build_datastore_streaming(train_data)
        
        results = self.evaluate_on_questions(test_data, output_file)
        
        print(f"\n✅ COMPLETE: {output_file}")
        return results
    
    def cleanup(self):
        # (This function is correct and unchanged)
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'faiss_index'):
            del self.faiss_index
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--k', type=int, default=1, help='Number of nearest neighbors (default: 1)')
    parser.add_argument('--upper-lambda', type=float, default=1.0)
    parser.add_argument('--lower-lambda', type=float, default=0.0)
    parser.add_argument('--distance-threshold', type=float, default=0.4)
    parser.add_argument('--batch-size', type=int, default=768) 
    parser.add_argument('--datastore-dir', default='datastore')
    parser.add_argument('--use-ivf', action='store_true')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild datastore even if it exists')
    
    args = parser.parse_args()
    
    generator = None
    try:
        generator = StreamingKNNLMGenerator(
            batch_size=args.batch_size,
            k=args.k,
            upper_lambda=args.upper_lambda,
            lower_lambda=args.lower_lambda,
            distance_threshold=args.distance_threshold,
            datastore_dir=args.datastore_dir,
            use_ivf=args.use_ivf,
            num_gpus=args.num_gpus
        )
        
        results = generator.run_complete_pipeline(
            args.train_file, 
            args.test_file, 
            args.output_file,
            force_rebuild=args.force_rebuild
        )
        print("\n✅ ALL DONE!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if generator:
            generator.cleanup()

if __name__ == "__main__":
    main()