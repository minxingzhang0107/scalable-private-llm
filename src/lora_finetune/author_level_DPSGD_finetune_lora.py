#!/usr/bin/env python
"""
Author-Level Differential Privacy LoRA Training
SIMPLE VERSION - High Epsilon, No Safety Checks
"""

import os
import json
import math
import torch
import numpy as np
import argparse
from collections import defaultdict, deque
import warnings
import tqdm.auto as tqdm
import time
import psutil
import gc
import sys
from pathlib import Path
import shutil

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)

# Set seed for reproducibility
set_seed(42)
warnings.filterwarnings("ignore")

def get_optimal_device():
    """Get the optimal device for training"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU (will be very slow)")
        return torch.device('cpu')

    best_device = 0
    max_free_memory = 0

    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        free_memory = total_memory - allocated_memory

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = i

    torch.cuda.set_device(best_device)
    device = torch.device(f'cuda:{best_device}')

    print(f"🎯 Selected device: {device} ({max_free_memory / 1e9:.1f}GB free)")
    return device

def validate_data_file(data_file):
    """Validate data file exists and has correct format"""
    print(f"📋 VALIDATING DATA FILE: {data_file}")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    file_size = os.path.getsize(data_file) / 1e6  # MB
    print(f"  📁 File size: {file_size:.1f}MB")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error - file may not be UTF-8: {e}")

    if not isinstance(data, list):
        raise ValueError("Data must be a list of examples")

    if len(data) == 0:
        raise ValueError("Data file is empty")

    print(f"  📊 Total examples: {len(data)}")
    return True

class MistralInstructionDataset:
    """Dataset for Mistral instruction tuning with author information"""

    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        question = example.get('question', '').strip()
        answer = example.get('answer', '').strip()

        if not question or not answer:
            question = "Empty question"
            answer = "Empty answer"

        # Mistral Instruct format
        instruction_prefix = f"<s>[INST] {question} [/INST] "
        full_text = instruction_prefix + f"{answer}</s>"

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = tokenized['input_ids'].squeeze()
        attention_mask = tokenized['attention_mask'].squeeze()
        labels = input_ids.clone()

        # Mask instruction tokens
        inst_end_tokens = self.tokenizer.encode("[/INST]", add_special_tokens=False)

        mask_until_idx = -1
        for k in range(len(input_ids) - len(inst_end_tokens) + 1):
            if torch.equal(input_ids[k:k + len(inst_end_tokens)], torch.tensor(inst_end_tokens)):
                mask_until_idx = k + len(inst_end_tokens)
                break

        if mask_until_idx != -1:
            if mask_until_idx < len(input_ids) and \
               self.tokenizer.decode([input_ids[mask_until_idx]]).strip() == '':
                mask_until_idx += 1
            labels[:mask_until_idx] = -100

        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'author': example.get('author', 'unknown')
        }

class SimpleLoRADPTrainer:
    """Simple LoRA DP Trainer - High Epsilon, No Safety Checks"""

    def __init__(self, model, tokenizer, examples_by_author, config, device):
        self.device = device
        self.tokenizer = tokenizer
        self.examples_by_author = examples_by_author
        self.authors = list(examples_by_author.keys())

        # Configuration
        self.target_epsilon = config['target_epsilon']
        self.target_delta = config['target_delta']
        self.max_grad_norm = config['max_grad_norm']
        self.user_sampling_rate = config['user_sampling_rate']
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.max_length = config['max_length']
        self.per_author_batch_size = config.get('per_author_batch_size', 20)

        # LoRA configuration
        lora_config = LoraConfig(
            r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 8),
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=config.get('lora_dropout', 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Setup model with LoRA
        self.model = get_peft_model(model, lora_config)
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Handle device placement
        if not hasattr(model, 'quantization_config') or model.quantization_config is None:
            self.model = self.model.to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.001,
            eps=1e-8
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=5, 
            gamma=0.9
        )

        # Simple noise multiplier calculation (high epsilon = minimal noise)
        self.noise_multiplier = self._calculate_simple_noise_multiplier()

        # Training state
        self.privacy_spent = 0.0
        self.step_count = 0

        # Statistics tracking
        self.loss_history = []
        self.step_times = deque(maxlen=50)

        # Per-author tracking
        self.author_gradient_norms = defaultdict(list)
        self.author_losses = defaultdict(list)
        self.author_sampling_counts = defaultdict(int)

        # Error tracking
        self.failed_authors = set()
        self.error_counts = defaultdict(int)
        self.consecutive_failures = 0

        self._print_initialization_summary()

    def _calculate_simple_noise_multiplier(self):
        """Simple noise calculation - high epsilon means minimal noise"""
        steps_per_epoch = max(1, int(len(self.authors) * self.user_sampling_rate))
        total_steps = steps_per_epoch * self.epochs
        
        try:
            # With high epsilon (1000), this will be very small
            sigma = math.sqrt(2 * total_steps * math.log(1.25 / self.target_delta)) / self.target_epsilon
            
            # Ensure minimum noise for numerical stability
            final_sigma = max(sigma, 1e-6)
            
            print(f"🔧 SIMPLE NOISE CALCULATION:")
            print(f"    Epsilon: {self.target_epsilon}")
            print(f"    Sigma: {final_sigma:.6f}")
            print(f"    Noise std will be: {self.max_grad_norm * final_sigma:.6f}")
            
            return final_sigma
            
        except (ValueError, ZeroDivisionError) as e:
            print(f"⚠️ Warning: Privacy calculation issue: {e}")
            return 1e-6  # Minimal noise

    def _print_initialization_summary(self):
        """Print initialization summary"""
        print(f"\n🎯 SIMPLE LORA DP TRAINER INITIALIZED:")
        print(f"  👥 Authors: {len(self.authors)}")
        print(f"  📊 Total Q&A pairs: {sum(len(examples) for examples in self.examples_by_author.values())}")
        print(f"  🎯 Sampling rate: {self.user_sampling_rate:.1%}")
        print(f"  📦 Examples per author: up to {self.per_author_batch_size}")
        print(f"  🔒 Privacy budget: ε={self.target_epsilon} (HIGH - minimal privacy)")
        print(f"  🔊 Noise multiplier: {self.noise_multiplier:.6f} (MINIMAL)")
        print(f"  🎚️ Noise std: {self.max_grad_norm * self.noise_multiplier:.6f}")
        print(f"  🧮 Learning rate: {self.learning_rate}")

    def _safe_author_gradient(self, author, step_num=0):
        """Compute AVERAGE gradient for author across all their examples"""
        print(f"  🔍 Processing author {author} (step {step_num})...")
        
        try:
            examples = self.examples_by_author[author]
            print(f"  📊 Author {author} has {len(examples)} examples")
            
            # Use up to per_author_batch_size examples
            if len(examples) > self.per_author_batch_size:
                examples = np.random.choice(examples, self.per_author_batch_size, replace=False).tolist()
                print(f"  ✂️ Using {len(examples)} examples for author {author}")

            author_gradient = None
            total_loss = 0.0
            processed_examples = 0

            # Process each example and accumulate gradients
            for i, example in enumerate(examples):
                print(f"    📝 Processing example {i+1}/{len(examples)} for author {author}")
                
                try:
                    # Clear gradients before each example
                    self.model.zero_grad()
                    
                    # Create dataset for single example
                    single_dataset = MistralInstructionDataset([example], self.tokenizer, self.max_length)
                    batch_data = single_dataset[0]
                    
                    # Move to device
                    input_ids = batch_data['input_ids'].unsqueeze(0).to(self.device, non_blocking=True)
                    attention_mask = batch_data['attention_mask'].unsqueeze(0).to(self.device, non_blocking=True)
                    labels = batch_data['labels'].unsqueeze(0).to(self.device, non_blocking=True)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                                
                    print(f"    📊 Loss computed: {loss.item():.4f}")

                    # Basic loss validation
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0 or loss.item() > 20:
                        print(f"    ❌ Invalid loss for author {author}, example {i}: {loss.item()}")
                        del input_ids, attention_mask, labels, outputs, loss
                        continue

                    # Backward pass
                    print(f"    ⬅️ Backward pass...")
                    loss.backward()
                    
                    # Basic gradient validation
                    nan_grads = False
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            if torch.isnan(param.grad).any():
                                nan_grads = True
                                break
                    
                    if nan_grads:
                        print(f"    ❌ NaN gradients detected for author {author}, example {i}")
                        self.model.zero_grad()
                        del input_ids, attention_mask, labels, outputs, loss
                        continue
                    
                    # Extract gradients for this example
                    example_gradients = []
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            grad = param.grad.detach().clone()
                            example_gradients.append(grad)
                        else:
                            example_gradients.append(None)
                    
                    # Accumulate gradients for averaging
                    if author_gradient is None:
                        author_gradient = example_gradients
                    else:
                        for j, grad in enumerate(example_gradients):
                            if grad is not None and author_gradient[j] is not None:
                                author_gradient[j] += grad
                    
                    total_loss += loss.item()
                    processed_examples += 1
                    
                    print(f"    ✅ Example {i+1} processed successfully")

                    # Clear intermediate tensors
                    del input_ids, attention_mask, labels, outputs, loss
                    
                except Exception as e:
                    print(f"    ❌ Error processing example {i} for author {author}: {e}")
                    continue

            print(f"  📊 Author {author} summary: {processed_examples}/{len(examples)} examples processed")
            
            if processed_examples == 0:
                print(f"  ❌ No examples processed for author {author}")
                return None, None

            # Average the accumulated gradients
            print(f"  📊 Averaging gradients over {processed_examples} examples...")
            for i, grad in enumerate(author_gradient):
                if grad is not None:
                    author_gradient[i] = grad / processed_examples

            # Clear gradients and memory
            self.model.zero_grad()
            torch.cuda.empty_cache()

            avg_loss = total_loss / processed_examples
            print(f"  ✅ Author {author} completed: avg_loss={avg_loss:.4f}")
            return author_gradient, avg_loss

        except Exception as e:
            self.failed_authors.add(author)
            print(f"  ❌ FATAL error processing author {author}: {e}")
            return None, None

    def _clip_gradients(self, author_gradients):
        """Simple gradient clipping"""
        try:
            total_norm_squared = 0.0

            for grad in author_gradients:
                if grad is not None and grad.numel() > 0:
                    total_norm_squared += grad.norm().item() ** 2

            total_norm = math.sqrt(total_norm_squared)

            if total_norm < 1e-10:
                return author_gradients, 0.0

            # Clip
            clip_coeff = min(1.0, self.max_grad_norm / total_norm)

            clipped_gradients = []
            for grad in author_gradients:
                if grad is not None:
                    clipped_gradients.append(grad * clip_coeff)
                else:
                    clipped_gradients.append(grad)

            return clipped_gradients, total_norm

        except Exception as e:
            print(f"⚠️ Clipping error: {e}")
            return author_gradients, 0.0

    def _add_minimal_noise(self, aggregated_gradient):
        """Add minimal noise (high epsilon)"""
        try:
            noisy_gradient = []
            noise_std = self.max_grad_norm * self.noise_multiplier

            print(f"    🔊 Adding minimal noise (std: {noise_std:.6f})...")

            for grad in aggregated_gradient:
                if grad is not None and grad.numel() > 0:
                    # Generate minimal noise
                    noise = torch.normal(
                        mean=0.0,
                        std=noise_std,
                        size=grad.shape,
                        device=grad.device,
                        dtype=grad.dtype
                    )
                    
                    # Apply noise
                    noisy_grad = grad + noise
                    noisy_gradient.append(noisy_grad)
                else:
                    noisy_gradient.append(grad)

            return noisy_gradient, noise_std

        except Exception as e:
            print(f"⚠️ Noise addition error: {e}")
            return aggregated_gradient, 0.0

    def training_step(self, step_num=0):
        """Simple training step"""
        step_start_time = time.time()
        
        print(f"\n🔍 STEP {step_num}: Starting simple LoRA DP step...")

        # Sample authors
        num_authors_to_sample = max(1, int(len(self.authors) * self.user_sampling_rate))
        available_authors = list(set(self.authors) - self.failed_authors)

        if len(available_authors) < num_authors_to_sample:
            num_authors_to_sample = len(available_authors)

        if num_authors_to_sample == 0:
            return {'success': False, 'error': 'no_available_authors'}

        sampled_authors = np.random.choice(available_authors, size=num_authors_to_sample, replace=False)
        print(f"🔍 STEP {step_num}: Selected authors: {sampled_authors[:3]}{'...' if len(sampled_authors) > 3 else ''}")

        # Process each author
        author_gradients = []
        total_loss = 0.0
        successful_authors = 0
        gradient_norms = []

        for i, author in enumerate(sampled_authors):
            print(f"🔍 STEP {step_num}: Processing author {i+1}/{len(sampled_authors)}: {author}")
            
            author_grads, author_loss = self._safe_author_gradient(author, step_num)

            if author_grads is None:
                print(f"❌ STEP {step_num}: Author {author} failed")
                continue

            # Clip gradients
            clipped_grads, grad_norm = self._clip_gradients(author_grads)

            if grad_norm == 0.0:
                print(f"❌ STEP {step_num}: Author {author} zero gradient norm")
                continue

            author_gradients.append(clipped_grads)
            total_loss += author_loss
            successful_authors += 1
            gradient_norms.append(grad_norm)
            
            print(f"✅ STEP {step_num}: Author {author} - Loss: {author_loss:.4f}, Grad norm: {grad_norm:.2f}")

        print(f"🔍 STEP {step_num}: Valid gradients from {successful_authors}/{len(sampled_authors)} authors")
        
        if successful_authors == 0:
            return {'success': False, 'error': 'no_valid_gradients'}

        # Aggregate author gradients (average across authors)
        print(f"🔍 STEP {step_num}: Aggregating {successful_authors} author gradients...")
        aggregated_gradient = []
        for param_idx in range(len(author_gradients[0])):
            if author_gradients[0][param_idx] is not None:
                # Average across authors
                param_grad = sum(grads[param_idx] for grads in author_gradients 
                                if grads[param_idx] is not None) / successful_authors
                aggregated_gradient.append(param_grad)
            else:
                aggregated_gradient.append(None)

        # Add minimal noise
        noisy_gradient, noise_std = self._add_minimal_noise(aggregated_gradient)

        # Apply gradients - NO SAFETY CHECKS
        print(f"🔍 STEP {step_num}: Applying gradients directly (no safety checks)...")
        with torch.no_grad():
            grad_applied_count = 0
            for param, grad in zip(self.model.parameters(), noisy_gradient):
                if param.requires_grad and grad is not None:
                    # Only check for NaN/Inf
                    if not (torch.isnan(grad).any() or torch.isinf(grad).any()):
                        param.grad = grad
                        grad_applied_count += 1

        print(f"🔍 STEP {step_num}: Applied gradients to {grad_applied_count} parameters")
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # Update statistics
        avg_loss = total_loss / successful_authors
        avg_grad_norm = np.mean(gradient_norms)
        step_time = time.time() - step_start_time

        self.loss_history.append(avg_loss)
        self.step_times.append(step_time)
        self.step_count += 1

        print(f"✅ STEP {step_num}: SUCCESS - Loss: {avg_loss:.4f}, Grad norm: {avg_grad_norm:.2f}")

        return {
            'success': True,
            'avg_loss': avg_loss,
            'avg_grad_norm': avg_grad_norm,
            'num_authors': successful_authors,
            'step_time': step_time,
            'noise_std': noise_std
        }

    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting Simple LoRA DP Training...")

        for epoch in range(self.epochs):
            print(f"\n📈 EPOCH {epoch + 1}/{self.epochs}")

            steps_per_epoch = max(5, int(len(self.authors) * self.user_sampling_rate))
            valid_steps = 0

            for step in range(steps_per_epoch):
                step_result = self.training_step(step_num=step)

                if step_result['success']:
                    valid_steps += 1
                    print(f"✅ STEP {step+1} SUCCESS:")
                    print(f"    Loss: {step_result['avg_loss']:.4f}")
                    print(f"    Authors processed: {step_result['num_authors']}")
                    print(f"    Gradient norm: {step_result['avg_grad_norm']:.2f}")
                    print(f"    Step time: {step_result['step_time']:.2f}s")
                    print(f"    Noise std: {step_result['noise_std']:.6f}")
                else:
                    print(f"❌ STEP {step+1} FAILED: {step_result.get('error', 'unknown')}")

            print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
            print(f"  ✅ Valid steps: {valid_steps}/{steps_per_epoch}")

        print(f"\n✅ Simple LoRA DP Training completed!")
        print(f"📊 Total successful steps: {self.step_count}")

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer"""
    print(f"🤖 Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # FIX: Use bfloat16 for numerical stability
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Model loaded")
    return model, tokenizer

def load_data(data_file):
    """Load and validate data"""
    validate_data_file(data_file)

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group by author
    examples_by_author = defaultdict(list)
    for example in data:
        if isinstance(example, dict) and all(field in example for field in ['question', 'answer', 'author']):
            examples_by_author[example['author']].append(example)

    print(f"✅ Data loaded: {len(examples_by_author)} authors, {len(data)} examples")
    return examples_by_author

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./simple_lora_dp_model')
    parser.add_argument('--epsilon', type=float, default=1000.0)  # High epsilon = minimal privacy
    parser.add_argument('--delta', type=float, default=1e-6)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--user-sampling-rate', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--per-author-batch-size', type=int, default=20)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=8)
    args = parser.parse_args()

    print("🚀 SIMPLE LORA DP WITH HIGH EPSILON")
    print(f"✅ Epsilon: {args.epsilon} (MINIMAL PRIVACY)")
    print("=" * 50)

    # Setup
    device = get_optimal_device()
    torch.cuda.empty_cache()
    
    # Load data and model
    examples_by_author = load_data(args.data_file)
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Config
    config = {
        'target_epsilon': args.epsilon,
        'target_delta': args.delta,
        'max_grad_norm': args.max_grad_norm,
        'user_sampling_rate': args.user_sampling_rate,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'per_author_batch_size': args.per_author_batch_size,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
    }

    # Train
    trainer = SimpleLoRADPTrainer(
        model=model,
        tokenizer=tokenizer,
        examples_by_author=examples_by_author,
        config=config,
        device=device
    )

    trainer.train()

    # Save
    print(f"\n💾 Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Model saved!")

if __name__ == "__main__":
    main()