#!/usr/bin/env python3
"""
LM-Only Generation: TASK 1a ONLY
No datastore building, just pure LM generation
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from typing import List, Tuple, Dict
import sys
import argparse
import os

class LMOnlyGenerator:
    """LM-only generator - Task 1a only"""
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("⚡ LM-ONLY GENERATOR (TASK 1a ONLY)")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Device: {self.device}")
        print(f"TASK: 1a-LM ONLY")
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
    
    # TASK 1a: LM-only generation - EXACTLY THE SAME
    def generate_lm_only(self, question: str, max_tokens: int = 75) -> str:
        """TASK 1a: Generate using LM only - EXACTLY THE SAME"""
        print(f"🤖 LM-ONLY GENERATION:")
        print(f"Question: {question}")
        
        # Start with prompt as text (similar to working version pattern)
        # private
        current_prompt = f"<s>[INST] Answer this question: {question} [/INST]"
        # public
        # current_prompt = f"<s>[INST] Question: {question}\n\nGive a direct and concise answer to the question above. [/INST]"
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
                # public: no; private: yes
                if len(generated_tokens) > 15:
                    recent_text = self.tokenizer.decode([next_token_id])
                    if any(punct in recent_text for punct in ['.', '!', '?']):
                        break
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"✅ LM Answer: '{generated_text.strip()}'")
        return generated_text.strip()
    
    def evaluate_on_questions(self, qa_pairs: List[dict], output_file: str):
        """Evaluate ONLY LM generation and save to JSON file"""
        print("\n🧪 LM-ONLY EVALUATION")
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
            
            # ONLY LM generation
            lm_answer = self.generate_lm_only(question)
            
            # Create result entry with same format as input, but replace answer
            result_entry = qa.copy()  # Keep all original fields
            result_entry['answer'] = lm_answer  # Replace with generated answer
            results.append(result_entry)
            
            print(f"{'='*80}")
        
        # Save results to JSON file
        print(f"\n💾 Saving results to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved successfully!")
        
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
    
    def run_complete_pipeline(self, test_file: str, output_file: str):
        """Run complete pipeline - NO DATASTORE BUILDING"""
        print(f"\n🎯 LM-ONLY PIPELINE (TASK 1a ONLY)")
        print(f"📁 Test: {test_file}")
        print(f"💾 Output: {output_file}")
        
        # Load data
        test_data = self.load_qa_data(test_file)
        
        if not test_data:
            print("❌ Missing test data file")
            return
        
        # Test ONLY LM generation - NO DATASTORE BUILDING
        results = self.evaluate_on_questions(test_data, output_file)
        
        print(f"\n🎯 PIPELINE COMPLETE!")
        print(f"   LM-only completed!")
        print(f"   Results saved to: {output_file}")
        return results
    
    def cleanup(self):
        """Cleanup"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--output-file', required=True, help='Output JSON file for generated answers')
    
    args = parser.parse_args()
    
    generator = None
    try:
        generator = LMOnlyGenerator()
        
        results = generator.run_complete_pipeline(args.test_file, args.output_file)
        print("\n✅ LM-ONLY TASK COMPLETED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if generator:
            generator.cleanup()

if __name__ == "__main__":
    main()