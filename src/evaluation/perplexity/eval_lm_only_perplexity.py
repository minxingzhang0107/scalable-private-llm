#!/usr/bin/env python3
"""
Perplexity Evaluation: Compare pretrained vs fine-tuned models
Evaluate perplexity on ground truth answers
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
import argparse
import os
import math

class PerplexityEvaluator:
    """Evaluate perplexity for both pretrained and fine-tuned models"""
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2", adapter_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("⚡ PERPLEXITY EVALUATOR")
        print("=" * 60)
        print(f"Base Model: {model_path}")
        if adapter_path:
            print(f"Fine-tuned Adapter: {adapter_path}")
        else:
            print("Fine-tuned Adapter: None")
        print(f"Device: {self.device}")
        print(f"TASK: Perplexity Evaluation")
        sys.stdout.flush()
        
        # Set deterministic mode
        self._set_deterministic()
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        sys.stdout.flush()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load pretrained model
        print("🤖 Loading pretrained model...")
        sys.stdout.flush()
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        # Load fine-tuned model if adapter provided
        self.finetuned_model = None
        if adapter_path and os.path.exists(adapter_path):
            print("🎯 Loading fine-tuned model...")
            sys.stdout.flush()
            # Load base model again for fine-tuned version
            base_model_ft = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=False
            )
            # Load fine-tuned adapter
            self.finetuned_model = PeftModel.from_pretrained(base_model_ft, adapter_path)
            self.finetuned_model.eval()
            for param in self.finetuned_model.parameters():
                param.requires_grad = False
            print("✅ Fine-tuned model loaded successfully")
        else:
            print("⚠️ No fine-tuned model will be evaluated")
        
        print("✅ Models loaded successfully")
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
    
    def calculate_perplexity(self, model, question: str, ground_truth_answer: str) -> float:
        """Calculate perplexity of ground truth answer given the question"""
        
        # PRESERVED: Create the same prompt format as in the original code
        prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        
        # PRESERVED: Tokenize full sequence together with space (same as original approach)
        full_sequence = f"{prompt} {ground_truth_answer}"
        full_encoding = self.tokenizer(full_sequence, return_tensors='pt', truncation=True, max_length=1024)
        full_input_ids = full_encoding['input_ids'].to(self.device)
        attention_mask = full_encoding['attention_mask'].to(self.device)
        
        # PRESERVED: Tokenize just the prompt to find where answer starts (same as original approach)
        prompt_encoding = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        # PRESERVED: Better boundary detection - verify tokenization consistency (same logic as original)
        # Check if the prompt portion matches when tokenized separately vs as part of full sequence
        prompt_from_full = full_input_ids[0, :prompt_length]
        prompt_separate = prompt_encoding['input_ids'][0].to(self.device)
        
        # PRESERVED: Same fallback logic, but FIXED the [/INST] token detection
        if not torch.equal(prompt_from_full, prompt_separate):
            # FIX: Properly handle multi-token [/INST] sequence
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
        
        # PRESERVED: Verify we have answer tokens (same check)
        if full_input_ids.shape[1] <= prompt_length:
            return float('inf')  # No answer tokens
        
        # PRESERVED: Get answer tokens (same calculation)
        answer_start_idx = prompt_length
        answer_end_idx = full_input_ids.shape[1]
        answer_length = answer_end_idx - answer_start_idx
        
        if answer_length == 0:
            return float('inf')
        
        # PRESERVED: Check for truncation (same check)
        if attention_mask[0, -1] == 0:  # Last token is masked (truncated)
            print(f"   Warning: Sequence may be truncated (length: {full_input_ids.shape[1]})")
        
        # PRESERVED: Calculate perplexity using one forward pass (same approach)
        with torch.no_grad():
            outputs = model(input_ids=full_input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # PRESERVED: Use log_softmax for numerical stability (same as original)
            log_probs = F.log_softmax(logits, dim=-1)
            
            total_log_prob = 0.0
            num_tokens = 0
            
            # PRESERVED: For each answer token, get its probability given previous context (same loop)
            for i in range(answer_length):
                target_pos = prompt_length + i
                context_pos = target_pos - 1
                
                if context_pos < 0 or target_pos >= full_input_ids.shape[1]:
                    continue
                
                # PRESERVED: Skip if attention mask indicates this token should be ignored (same check)
                if attention_mask[0, target_pos] == 0:
                    continue
                
                # Get the target token ID
                target_token_id = full_input_ids[0, target_pos].item()
                
                # FIX: Add bounds checking for token ID (safety check)
                if target_token_id < 0 or target_token_id >= log_probs.shape[1]:
                    continue
                
                # PRESERVED: Get log probability directly from log_softmax (same method)
                target_log_prob = log_probs[context_pos, target_token_id].item()
                
                # PRESERVED: Proper error handling - skip non-finite values (same logic)
                if math.isfinite(target_log_prob):
                    total_log_prob += target_log_prob
                    num_tokens += 1
        
        if num_tokens == 0:
            return float('inf')
        
        # PRESERVED: Calculate perplexity: exp(-1/N * sum(log(p_i))) (exact same formula)
        avg_log_prob = total_log_prob / num_tokens
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def evaluate_perplexity_on_dataset(self, qa_pairs: List[dict]) -> Dict[str, float]:
        """Evaluate perplexity on the entire dataset"""
        print("\n🧪 PERPLEXITY EVALUATION")
        print("=" * 80)
        sys.stdout.flush()
        
        pretrained_perplexities = []
        finetuned_perplexities = []
        
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa['question']
            ground_truth = qa['answer']
            
            print(f"\n{'='*10} QUESTION {qa_idx + 1}/{len(qa_pairs)} {'='*10}")
            print(f"Q: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"{'='*50}")
            
            # FIX: Add basic error handling for individual questions
            try:
                # Evaluate pretrained model
                print("🤖 Evaluating pretrained model...")
                pretrained_ppl = self.calculate_perplexity(self.pretrained_model, question, ground_truth)
                pretrained_perplexities.append(pretrained_ppl)
                print(f"   Pretrained perplexity: {pretrained_ppl:.4f}")
                
                # Evaluate fine-tuned model if available
                if self.finetuned_model is not None:
                    print("🎯 Evaluating fine-tuned model...")
                    finetuned_ppl = self.calculate_perplexity(self.finetuned_model, question, ground_truth)
                    finetuned_perplexities.append(finetuned_ppl)
                    print(f"   Fine-tuned perplexity: {finetuned_ppl:.4f}")
                    
            except Exception as e:
                print(f"   Error evaluating question {qa_idx + 1}: {e}")
                # Add inf values to maintain list alignment
                pretrained_perplexities.append(float('inf'))
                if self.finetuned_model is not None:
                    finetuned_perplexities.append(float('inf'))
            
            print(f"{'='*80}")
            
            # FIX: Add occasional memory cleanup to prevent OOM
            if (qa_idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # PRESERVED: Calculate average perplexities (same calculation)
        # FIX: Handle potential inf values in averaging
        def safe_mean(values):
            finite_values = [v for v in values if math.isfinite(v)]
            return np.mean(finite_values) if finite_values else float('inf')
        
        avg_pretrained_ppl = safe_mean(pretrained_perplexities)
        avg_finetuned_ppl = safe_mean(finetuned_perplexities) if finetuned_perplexities else None
        
        results = {
            'pretrained_perplexity': avg_pretrained_ppl,
            'finetuned_perplexity': avg_finetuned_ppl,
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
        # FIX: Add handling for JSON decode errors
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return []
    
    def run_evaluation(self, test_file: str):
        """Run complete perplexity evaluation"""
        print(f"\n🎯 PERPLEXITY EVALUATION PIPELINE")
        print(f"📁 Test: {test_file}")
        
        # Load data
        test_data = self.load_qa_data(test_file)
        
        if not test_data:
            print("❌ Missing test data file")
            return None
        
        # Evaluate perplexity
        results = self.evaluate_perplexity_on_dataset(test_data)
        
        # Print final results
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   📊 Dataset: {len(test_data)} questions")
        print(f"   🤖 Pretrained Model Perplexity: {results['pretrained_perplexity']:.4f}")
        if results['finetuned_perplexity'] is not None:
            print(f"   🎯 Fine-tuned Model Perplexity: {results['finetuned_perplexity']:.4f}")
            improvement = results['pretrained_perplexity'] - results['finetuned_perplexity']
            print(f"   📈 Improvement: {improvement:.4f} (lower is better)")
        else:
            print(f"   🎯 Fine-tuned Model: Not evaluated")
        
        return results
    
    def cleanup(self):
        """Cleanup"""
        if hasattr(self, 'pretrained_model'):
            del self.pretrained_model
        if hasattr(self, 'finetuned_model') and self.finetuned_model is not None:
            del self.finetuned_model
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', required=True, help='Test JSON file with questions and answers')
    parser.add_argument('--adapter-path', default=None, help='Path to fine-tuned LoRA adapter (optional)')
    
    args = parser.parse_args()
    
    evaluator = None
    try:
        evaluator = PerplexityEvaluator(adapter_path=args.adapter_path)
        
        results = evaluator.run_evaluation(args.test_file)
        
        if results:
            print("\n✅ PERPLEXITY EVALUATION COMPLETED!")
            print(f"   🤖 Pretrained: {results['pretrained_perplexity']:.4f}")
            if results['finetuned_perplexity'] is not None:
                print(f"   🎯 Fine-tuned: {results['finetuned_perplexity']:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if evaluator:
            evaluator.cleanup()

if __name__ == "__main__":
    main()