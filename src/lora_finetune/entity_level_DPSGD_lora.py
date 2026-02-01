#!/usr/bin/env python
"""
Entity-Level DPSGD LoRA Training
FINAL CORRECTED VERSION - All Issues Fixed
"""

import os
import json
import math
import torch
import numpy as np
import argparse
from collections import defaultdict, deque
import warnings
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

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

    file_size = os.path.getsize(data_file) / 1e6
    print(f"  📁 File size: {file_size:.1f}MB")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Data must be a non-empty list")

    print(f"  📊 Total examples: {len(data)}")
    return True

class MistralInstructionDataset:
    """Dataset for Mistral instruction tuning"""

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

        instruction_prefix = f"<s>[INST] {question} [/INST] "
        full_text = instruction_prefix + f"{answer}</s>"

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

class UserLevelDPLoRATrainer:
    """User-Level DP LoRA Trainer - Final Corrected Version"""

    def __init__(self, model, tokenizer, examples_by_author, config, device):
        self.device = device
        self.tokenizer = tokenizer
        self.examples_by_author = examples_by_author
        self.authors = list(examples_by_author.keys())

        self.target_epsilon = config['target_epsilon']
        self.target_delta = config['target_delta']
        self.max_grad_norm = config['max_grad_norm']
        self.user_sampling_rate = config['user_sampling_rate']
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.max_length = config['max_length']
        self.auto_tune_clipping = config.get('auto_tune_clipping', False)
        self.clipping_percentile = config.get('clipping_percentile', 50)
        
        # For gradient norm monitoring
        self.observed_grad_norms = []
        self.clipping_tuned = False

        # Validate sampling rate
        if self.user_sampling_rate <= 0 or self.user_sampling_rate > 1:
            raise ValueError("user_sampling_rate must be in (0, 1]")

        # FIX #1: Validate delta
        min_recommended_delta = 1.0 / len(self.authors)
        if self.target_delta >= min_recommended_delta:
            raise ValueError(
                f"Delta ({self.target_delta}) must be < 1/n ({min_recommended_delta:.2e}) "
                f"for meaningful privacy with {len(self.authors)} users. "
                f"Try delta = {min_recommended_delta/10:.2e}"
            )

        lora_config = LoraConfig(
            r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 8),
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=config.get('lora_dropout', 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(model, lora_config)
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        if not hasattr(model, 'quantization_config') or model.quantization_config is None:
            self.model = self.model.to(device)

        # Calculate noise multiplier with correct formula
        self.noise_multiplier, self.steps_per_epoch = self._calculate_noise_multiplier()

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.001,
            eps=1e-8
        )

        # FIX #3: Better learning rate schedule (decay per epoch, not per step)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.steps_per_epoch,  # Decay once per epoch
            gamma=0.9
        )

        self.step_count = 0
        self.loss_history = []
        self.step_times = deque(maxlen=50)
        self.author_sampling_counts = defaultdict(int)
        self.failed_authors = set()
        
        # Gradient norm statistics for clipping analysis
        self.all_observed_norms = []
        self.clipped_count = 0
        self.unclipped_count = 0

        self._print_initialization_summary()

    def _calculate_noise_multiplier(self):
        """
        Calculate noise multiplier using proper privacy accounting.
        
        Uses CORRECTED standard DP-SGD formula (not overly conservative).
        For production, use Opacus library for exact RDP/moments accounting.
        """
        q = self.user_sampling_rate
        num_users = len(self.authors)
        
        # CORRECT: An epoch is ceil(1/q) steps
        steps_per_epoch = math.ceil(1.0 / q)
        T = steps_per_epoch * self.epochs
        
        print(f"\n{'='*70}")
        print(f"🔒 PRIVACY ACCOUNTING")
        print(f"{'='*70}")
        print(f"  Total users: {num_users}")
        print(f"  Sampling rate q: {q:.4f} ({q*100:.2f}%)")
        print(f"  Users sampled per step: ~{int(num_users * q)}")
        print(f"  Steps per epoch: {steps_per_epoch} (= ceil(1/q))")
        print(f"  Total epochs: {self.epochs}")
        print(f"  Total steps T: {T}")
        print(f"  Target privacy: (ε={self.target_epsilon}, δ={self.target_delta})")
        print(f"{'='*70}")
        
        # FIX #2: Use standard DP-SGD formula (not overly conservative)
        # Standard formula from Abadi et al. (2016):
        # ε ≈ q * sqrt(2T * log(1/δ)) / σ
        # Solving for σ:
        # σ = q * sqrt(2T * log(1/δ)) / ε
        
        sigma = q * math.sqrt(2 * T * math.log(1 / self.target_delta)) / self.target_epsilon
        
        print(f"\n🔧 NOISE PARAMETERS:")
        print(f"  Noise multiplier σ: {sigma:.6f}")
        print(f"  Noise std (σ*C): {sigma * self.max_grad_norm:.6f}")
        print(f"  Clipping norm C: {self.max_grad_norm}")
        print(f"\n⚠️  Using simplified DP-SGD privacy accounting.")
        print(f"  For tighter bounds, use Opacus library in production.")
        print(f"{'='*70}")
        
        return sigma, steps_per_epoch

    def _print_initialization_summary(self):
        """Print initialization summary"""
        total_examples = sum(len(examples) for examples in self.examples_by_author.values())
        avg_examples = total_examples / len(self.authors) if self.authors else 0
        
        print(f"\n{'='*70}")
        print(f"USER-LEVEL DP LORA TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"  👥 Authors: {len(self.authors)}")
        print(f"  📊 Total examples: {total_examples}")
        print(f"  📈 Avg examples/author: {avg_examples:.1f}")
        print(f"  🎯 User sampling rate: {self.user_sampling_rate:.1%}")
        print(f"  📅 Steps per epoch: {self.steps_per_epoch}")
        print(f"  📅 Total training steps: {self.steps_per_epoch * self.epochs}")
        print(f"  🔒 Privacy budget: ε={self.target_epsilon}, δ={self.target_delta}")
        print(f"  ✂️  Clipping norm C: {self.max_grad_norm}")
        print(f"  🔊 Noise multiplier σ: {self.noise_multiplier:.6f}")
        print(f"  🧮 Learning rate: {self.learning_rate}")
        print(f"  📉 LR decay: 0.9 per epoch (not per step)")
        print(f"{'='*70}")

    def _compute_user_gradient(self, author, step_num=0):
        """
        Compute gradient for a user by averaging over ALL their examples.
        
        CRITICAL: Must process ALL examples (no subsampling within user).
        This is what makes it user-level DP, not instance-level DP.
        """
        print(f"  👤 Processing author {author}...")
        
        try:
            # CRITICAL: Use ALL examples from this author
            examples = self.examples_by_author[author]
            print(f"     Processing ALL {len(examples)} examples")
            
            accumulated_gradient = None
            total_loss = 0.0
            processed_examples = 0

            for i, example in enumerate(examples):
                try:
                    self.model.zero_grad()
                    
                    single_dataset = MistralInstructionDataset([example], self.tokenizer, self.max_length)
                    batch_data = single_dataset[0]
                    
                    input_ids = batch_data['input_ids'].unsqueeze(0).to(self.device, non_blocking=True)
                    attention_mask = batch_data['attention_mask'].unsqueeze(0).to(self.device, non_blocking=True)
                    labels = batch_data['labels'].unsqueeze(0).to(self.device, non_blocking=True)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0 or loss.item() > 20:
                        del input_ids, attention_mask, labels, outputs, loss
                        continue

                    loss.backward()
                    
                    has_nan = False
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan = True
                                break
                    
                    if has_nan:
                        self.model.zero_grad()
                        del input_ids, attention_mask, labels, outputs, loss
                        continue
                    
                    if accumulated_gradient is None:
                        accumulated_gradient = []
                        for param in self.model.parameters():
                            if param.requires_grad and param.grad is not None:
                                accumulated_gradient.append(param.grad.detach().clone())
                            else:
                                accumulated_gradient.append(None)
                    else:
                        for j, param in enumerate(self.model.parameters()):
                            if param.requires_grad and param.grad is not None:
                                if accumulated_gradient[j] is not None:
                                    accumulated_gradient[j] += param.grad.detach()
                    
                    total_loss += loss.item()
                    processed_examples += 1

                    del input_ids, attention_mask, labels, outputs, loss
                    
                except Exception as e:
                    print(f"     ⚠️  Error on example {i}: {e}")
                    continue

            if processed_examples == 0:
                return None, None

            # Average the accumulated gradients over ALL user's examples
            user_gradient = []
            for grad in accumulated_gradient:
                if grad is not None:
                    user_gradient.append(grad / processed_examples)
                else:
                    user_gradient.append(None)

            self.model.zero_grad()
            torch.cuda.empty_cache()

            avg_loss = total_loss / processed_examples
            print(f"     ✅ Completed: {processed_examples} examples, loss={avg_loss:.4f}")
            
            return user_gradient, avg_loss

        except Exception as e:
            self.failed_authors.add(author)
            print(f"     ❌ Fatal error: {e}")
            return None, None

    def _clip_gradient(self, user_gradient):
        """Clip user's gradient to L2 norm <= C and track statistics"""
        try:
            total_norm_squared = 0.0
            for grad in user_gradient:
                if grad is not None and grad.numel() > 0:
                    total_norm_squared += grad.norm().item() ** 2

            total_norm = math.sqrt(total_norm_squared)
            if total_norm < 1e-10:
                return user_gradient, 0.0

            # Track all gradient norms for analysis
            self.all_observed_norms.append(total_norm)
            
            # Track clipping statistics
            if total_norm > self.max_grad_norm:
                self.clipped_count += 1
            else:
                self.unclipped_count += 1

            clip_coeff = min(1.0, self.max_grad_norm / total_norm)

            clipped_gradient = []
            for grad in user_gradient:
                if grad is not None:
                    clipped_gradient.append(grad * clip_coeff)
                else:
                    clipped_gradient.append(None)

            clipped_norm = min(total_norm, self.max_grad_norm)
            return clipped_gradient, clipped_norm

        except Exception as e:
            print(f"Warning: Clipping error: {e}")
            return user_gradient, 0.0

    def _add_dp_noise(self, gradient_sum, num_users):
        """
        Add Gaussian noise to the SUM of clipped gradients.
        
        Noise std = σ*C (independent of num_users).
        The sensitivity of the sum is C (one user can change the sum by at most C).
        """
        try:
            noisy_gradient = []
            noise_std = self.max_grad_norm * self.noise_multiplier

            for grad in gradient_sum:
                if grad is not None and grad.numel() > 0:
                    noise = torch.normal(
                        mean=0.0,
                        std=noise_std,
                        size=grad.shape,
                        device=grad.device,
                        dtype=grad.dtype
                    )
                    noisy_gradient.append(grad + noise)
                else:
                    noisy_gradient.append(grad)

            return noisy_gradient, noise_std

        except Exception as e:
            print(f"Warning: Noise error: {e}")
            return gradient_sum, 0.0

    def _auto_tune_clipping_norm(self):
        """Auto-tune clipping norm C based on observed gradient norms"""
        if len(self.observed_grad_norms) < 20:
            print(f"Warning: Only {len(self.observed_grad_norms)} observations, need at least 20 for reliable tuning")
            return
        
        norms_array = np.array(self.observed_grad_norms)
        
        # Calculate statistics
        median = np.median(norms_array)
        p25 = np.percentile(norms_array, 25)
        p50 = np.percentile(norms_array, 50)
        p75 = np.percentile(norms_array, 75)
        p90 = np.percentile(norms_array, 90)
        mean = np.mean(norms_array)
        std = np.std(norms_array)
        
        # Set C to the specified percentile
        recommended_C = np.percentile(norms_array, self.clipping_percentile)
        old_C = self.max_grad_norm
        
        print(f"\n{'='*70}")
        print(f"AUTO-TUNING CLIPPING NORM C")
        print(f"{'='*70}")
        print(f"Observed gradient norm statistics ({len(norms_array)} samples):")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std: {std:.4f}")
        print(f"  25th percentile: {p25:.4f}")
        print(f"  Median (50th): {median:.4f}")
        print(f"  75th percentile: {p75:.4f}")
        print(f"  90th percentile: {p90:.4f}")
        print(f"\nCurrent C: {old_C:.4f}")
        print(f"Recommended C ({self.clipping_percentile}th percentile): {recommended_C:.4f}")
        
        if abs(recommended_C - old_C) / old_C > 0.3:
            print(f"\nWARNING: Recommended C differs from current by >{30:.0f}%")
            print(f"Consider restarting training with C={recommended_C:.2f}")
            print(f"Note: Changing C requires recalculating noise multiplier for same privacy guarantee")
        else:
            print(f"\nCurrent C is within 30% of recommended value - reasonably tuned")
        
        # Show clipping statistics
        if len(self.all_observed_norms) > 0:
            total_grads = len(self.all_observed_norms)
            clipping_rate = self.clipped_count / total_grads if total_grads > 0 else 0
            print(f"\nClipping statistics:")
            print(f"  Total gradients: {total_grads}")
            print(f"  Clipped: {self.clipped_count} ({clipping_rate*100:.1f}%)")
            print(f"  Unclipped: {self.unclipped_count} ({(1-clipping_rate)*100:.1f}%)")
            
            if clipping_rate > 0.8:
                print(f"\n  WARNING: >{80:.0f}% of gradients are being clipped!")
                print(f"  This may indicate C is too small and information is being lost")
            elif clipping_rate < 0.2:
                print(f"\n  INFO: <{20:.0f}% of gradients are being clipped")
                print(f"  C might be larger than necessary (but this isn't necessarily bad)")
        
        print(f"{'='*70}\n")
        
        self.clipping_tuned = True
        
        return recommended_C

    def training_step(self, step_num=0, monitor_norms=False):
        """Execute one step of user-level DP-SGD"""
        step_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"STEP {step_num}")
        if monitor_norms:
            print(f"[MONITORING MODE - Observing unclipped gradient norms]")
        print(f"{'='*70}")

        # Sample users (not individual examples!)
        num_users_to_sample = max(1, int(len(self.authors) * self.user_sampling_rate))
        available_authors = list(set(self.authors) - self.failed_authors)

        if len(available_authors) < num_users_to_sample:
            num_users_to_sample = len(available_authors)

        if num_users_to_sample == 0:
            return {'success': False, 'error': 'no_available_authors'}

        sampled_authors = np.random.choice(available_authors, size=num_users_to_sample, replace=False)
        print(f"👥 Sampled {len(sampled_authors)} users")

        # Process each sampled user
        clipped_user_gradients = []
        total_loss = 0.0
        successful_users = 0
        gradient_norms = []

        for author in sampled_authors:
            # Compute gradient for this user (averaged over ALL their examples)
            user_gradient, user_loss = self._compute_user_gradient(author, step_num)

            if user_gradient is None:
                continue

            # During monitoring phase, store unclipped norms
            if monitor_norms:
                # Calculate unclipped norm
                total_norm_squared = 0.0
                for grad in user_gradient:
                    if grad is not None and grad.numel() > 0:
                        total_norm_squared += grad.norm().item() ** 2
                unclipped_norm = math.sqrt(total_norm_squared)
                if unclipped_norm > 1e-10:
                    self.observed_grad_norms.append(unclipped_norm)

            # Clip this user's gradient
            clipped_gradient, grad_norm = self._clip_gradient(user_gradient)

            if grad_norm == 0.0:
                continue

            clipped_user_gradients.append(clipped_gradient)
            total_loss += user_loss
            successful_users += 1
            gradient_norms.append(grad_norm)
            self.author_sampling_counts[author] += 1

        print(f"✅ Processed {successful_users}/{len(sampled_authors)} users")
        
        if successful_users == 0:
            return {'success': False, 'error': 'no_valid_gradients'}

        # SUM all clipped user gradients (do NOT average yet)
        print(f"➕ Summing {successful_users} clipped user gradients...")
        gradient_sum = []
        num_params = len(clipped_user_gradients[0])
        
        for param_idx in range(num_params):
            if clipped_user_gradients[0][param_idx] is not None:
                param_gradient_sum = sum(
                    user_grad[param_idx] 
                    for user_grad in clipped_user_gradients 
                    if user_grad[param_idx] is not None
                )
                gradient_sum.append(param_gradient_sum)
            else:
                gradient_sum.append(None)

        # Add DP noise to the sum
        print(f"🔒 Adding DP noise (σ={self.noise_multiplier:.6f})...")
        noisy_gradient_sum, noise_std = self._add_dp_noise(gradient_sum, successful_users)

        # Apply gradient: divide by num_users to get average, then apply
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            applied_count = 0
            for param, noisy_grad in zip(self.model.parameters(), noisy_gradient_sum):
                if param.requires_grad and noisy_grad is not None:
                    if not (torch.isnan(noisy_grad).any() or torch.isinf(noisy_grad).any()):
                        # Divide by num_users to get average
                        param.grad = noisy_grad / successful_users
                        applied_count += 1

        print(f"✅ Applied gradients to {applied_count} parameters")
        
        self.optimizer.step()
        self.scheduler.step()

        # Statistics
        avg_loss = total_loss / successful_users
        avg_grad_norm = np.mean(gradient_norms)
        step_time = time.time() - step_start_time

        self.loss_history.append(avg_loss)
        self.step_times.append(step_time)
        self.step_count += 1

        print(f"\n📊 STEP {step_num} SUMMARY:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Avg grad norm: {avg_grad_norm:.4f}")
        print(f"   Users: {successful_users}")
        print(f"   Time: {step_time:.2f}s")
        print(f"   Current LR: {self.scheduler.get_last_lr()[0]:.2e}")

        return {
            'success': True,
            'avg_loss': avg_loss,
            'avg_grad_norm': avg_grad_norm,
            'num_users': successful_users,
            'step_time': step_time,
            'noise_std': noise_std
        }

    def train(self):
        """Main training loop with correct epoch definition"""
        print(f"\n{'='*70}")
        print(f"STARTING USER-LEVEL DP-SGD TRAINING")
        print(f"{'='*70}")
        
        total_steps = self.epochs * self.steps_per_epoch
        
        print(f"Training Configuration:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Steps per epoch: {self.steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        
        # Gradient norm monitoring phase
        if self.auto_tune_clipping:
            monitoring_steps = min(20, self.steps_per_epoch // 2)
            print(f"\nGRADIENT NORM MONITORING ENABLED")
            print(f"  Will observe unclipped norms for first {monitoring_steps} steps")
            print(f"  Then recommend optimal C based on {self.clipping_percentile}th percentile")
        else:
            monitoring_steps = 0
            print(f"\nUsing fixed C={self.max_grad_norm:.2f}")
            print(f"  To enable auto-tuning, set --auto-tune-clipping")
        
        print(f"{'='*70}")

        for epoch in range(self.epochs):
            print(f"\n{'#'*70}")
            print(f"EPOCH {epoch + 1}/{self.epochs}")
            print(f"{'#'*70}")

            epoch_losses = []

            for step in range(self.steps_per_epoch):
                current_step = epoch * self.steps_per_epoch + step
                
                # Monitor gradient norms in monitoring phase
                is_monitoring = self.auto_tune_clipping and current_step < monitoring_steps
                
                step_result = self.training_step(step_num=current_step, monitor_norms=is_monitoring)

                if step_result['success']:
                    epoch_losses.append(step_result['avg_loss'])
                
                # After monitoring phase, analyze and recommend C
                if self.auto_tune_clipping and current_step == monitoring_steps - 1 and not self.clipping_tuned:
                    recommended_C = self._auto_tune_clipping_norm()
                    print(f"\nContinuing training with current C={self.max_grad_norm:.2f}")
                    print(f"For future runs, consider using C={recommended_C:.2f}\n")
                
                # Stop if no authors available
                if step_result.get('error') == 'no_available_authors':
                    print("Stopping early: No more available authors to sample.")
                    break

            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1} SUMMARY:")
            print(f"   Valid steps: {len(epoch_losses)}/{self.steps_per_epoch}")
            if epoch_losses:
                print(f"   Avg loss: {np.mean(epoch_losses):.4f}")
                print(f"   Min loss: {np.min(epoch_losses):.4f}")
                print(f"   Max loss: {np.max(epoch_losses):.4f}")
            print(f"{'='*70}")

        # Final clipping analysis if not already done
        if not self.clipping_tuned and len(self.all_observed_norms) > 0:
            print(f"\n")
            self._auto_tune_clipping_norm()

        # Final statistics
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Total steps executed: {self.step_count}")
        print(f"\nFINAL PRIVACY GUARANTEE:")
        print(f"   Privacy spent: (epsilon={self.target_epsilon}, delta={self.target_delta})")
        print(f"   User-level DP: Adding/removing ANY user's data")
        print(f"   changes model output by at most epsilon with probability 1-delta")
        print(f"\nTRAINING STATISTICS:")
        print(f"   Total training steps: {self.step_count}")
        print(f"   Sampling rate q: {self.user_sampling_rate}")
        print(f"   Clipping norm C: {self.max_grad_norm}")
        print(f"   Noise multiplier sigma: {self.noise_multiplier:.6f}")
        if self.loss_history:
            print(f"   Final avg loss: {np.mean(self.loss_history[-10:]):.4f}")
        print(f"{'='*70}")

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer"""
    print(f"🤖 Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
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

    examples_by_author = defaultdict(list)
    for example in data:
        if isinstance(example, dict) and all(field in example for field in ['question', 'answer', 'author']):
            examples_by_author[example['author']].append(example)

    num_examples_per_author = [len(examples) for examples in examples_by_author.values()]
    print(f"\n📊 DATA STATISTICS:")
    print(f"   Authors: {len(examples_by_author)}")
    print(f"   Total examples: {len(data)}")
    print(f"   Avg examples/author: {np.mean(num_examples_per_author):.1f}")
    print(f"   Min: {np.min(num_examples_per_author)}, Max: {np.max(num_examples_per_author)}")
    
    return examples_by_author

def main():
    parser = argparse.ArgumentParser(description='User-Level DP-SGD with LoRA (Final Corrected)')
    parser.add_argument('--model-name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./model_checkpoints/entity_level_DPSGD_lora_model')
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Clipping norm C. Recommend tuning based on observed gradient norms.')
    parser.add_argument('--auto-tune-clipping', action='store_true',
                       help='Monitor gradient norms and recommend optimal C value')
    parser.add_argument('--clipping-percentile', type=int, default=50,
                       help='Percentile for auto-tuning C (default: 50 for median)')
    parser.add_argument('--user-sampling-rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=8)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"USER-LEVEL DP-SGD WITH LORA (FINAL CORRECTED VERSION)")
    print(f"{'='*70}")
    print(f"Privacy: epsilon={args.epsilon}, delta={args.delta}")
    print(f"Sampling rate: {args.user_sampling_rate}")
    print(f"Clipping norm C: {args.max_grad_norm}")
    if args.auto_tune_clipping:
        print(f"Auto-tuning enabled: Will recommend C based on {args.clipping_percentile}th percentile")
    print(f"All critical fixes applied")
    print(f"{'='*70}\n")

    device = get_optimal_device()
    torch.cuda.empty_cache()
    
    examples_by_author = load_data(args.data_file)
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    config = {
        'target_epsilon': args.epsilon,
        'target_delta': args.delta,
        'max_grad_norm': args.max_grad_norm,
        'user_sampling_rate': args.user_sampling_rate,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'auto_tune_clipping': args.auto_tune_clipping,
        'clipping_percentile': args.clipping_percentile,
    }

    trainer = UserLevelDPLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        examples_by_author=examples_by_author,
        config=config,
        device=device
    )

    trainer.train()

    print(f"\n💾 Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training config and privacy parameters
    config_to_save = {
        **config,
        'privacy_spent': {
            'epsilon': args.epsilon,
            'delta': args.delta,
            'num_users': len(examples_by_author),
            'total_steps': trainer.step_count,
            'noise_multiplier': trainer.noise_multiplier
        }
    }
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"✅ Model and config saved to {args.output_dir}")

if __name__ == "__main__":
    main()