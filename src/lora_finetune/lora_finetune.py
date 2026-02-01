#!/usr/bin/env python3
"""
lora_finetune.py - Robust LoRA Fine-tuning for Mistral 7B
Instruction tuning with improved data handling and explicit padding within dataset.
"""

import torch
import numpy as np
import os
import json
import argparse
import logging
import warnings
from datetime import datetime
from tqdm import tqdm

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling # Still useful for label shifting/final checks
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)
from torch.utils.data import Dataset

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*safetensors.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Your input ran out of data.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MistralInstructionDataset(Dataset):
    """Dataset for Mistral 7B instruction tuning with proper formatting and label masking."""

    def __init__(self, qa_pairs, tokenizer, max_length=512):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        logger.info(f"🎯 Processing {len(qa_pairs)} Q&A pairs for dataset...")

        for i, qa in enumerate(tqdm(qa_pairs, desc="Preparing dataset examples")):
            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()

            if not question or not answer:
                logger.warning(f"Skipping example {i} due to missing question or answer.")
                continue

            try:
                # Mistral Instruct format: <s>[INST] Instruction [/INST] Model response</s>
                instruction_prefix = f"<s>[INST] {question} [/INST] " # Note the space after [/INST]
                full_text_with_answer = instruction_prefix + f"{answer}</s>"

                # Tokenize the full sequence with truncation, but NO PADDING yet.
                tokenized_full = self.tokenizer(
                    full_text_with_answer,
                    truncation=True,
                    max_length=self.max_length, # Use self.max_length here
                    padding=False,             # IMPORTANT: No padding at this stage
                    return_tensors=None,       # Return lists of integers
                    add_special_tokens=True
                )

                input_ids = tokenized_full['input_ids']
                attention_mask = tokenized_full['attention_mask']
                labels = list(input_ids).copy() # Labels start as a copy of input_ids

                # Find the token IDs for [/INST] to precisely locate the split point
                inst_end_tokens_sequence = self.tokenizer.encode("[/INST]", add_special_tokens=False)

                mask_until_idx = -1
                # Iterate through input_ids to find the sequence of [/INST] tokens
                for k in range(len(input_ids) - len(inst_end_tokens_sequence) + 1):
                    if input_ids[k : k + len(inst_end_tokens_sequence)] == inst_end_tokens_sequence:
                        mask_until_idx = k + len(inst_end_tokens_sequence)
                        break

                if mask_until_idx != -1:
                    # After [/INST], there's typically a space token (ID 28705 or similar)
                    # that should also be masked if it precedes the answer.
                    if mask_until_idx < len(input_ids) and \
                       self.tokenizer.decode([input_ids[mask_until_idx]]).strip() == '':
                        mask_until_idx += 1 # Mask the trailing space too

                    # Apply masking to labels: set instruction tokens to -100
                    for j in range(min(mask_until_idx, len(labels))):
                        labels[j] = -100

                    if len(input_ids) > 0 and len(input_ids) == len(labels):
                        # Store as lists. Padding will be handled in __getitem__
                        self.examples.append({
                            'input_ids': input_ids,
                            'labels': labels,
                            'attention_mask': attention_mask
                        })
                    else:
                        logger.warning(f"Example {i}: Mismatch in lengths or empty input_ids after processing. Input_ids: {len(input_ids)}, Labels: {len(labels)}. Skipping.")
                else:
                    logger.warning(f"Example {i}: Could not find '[/INST]' sequence in tokenized text. This may indicate an issue with prompt formatting or tokenizer. Skipping.")

            except Exception as e:
                logger.warning(f"Error processing example {i}: {e}", exc_info=True)
                continue

        logger.info(f"✅ Successfully processed {len(self.examples)} examples for training.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Explicitly pad the sequences to max_length here
        # This ensures all tensors passed to the DataLoader/collator have consistent dimensions
        current_length = len(example['input_ids'])
        if current_length < self.max_length:
            padding_length = self.max_length - current_length
            # Pad input_ids and labels with tokenizer.pad_token_id (or 0 for labels)
            # Pad attention_mask with 0
            padded_input_ids = example['input_ids'] + [self.tokenizer.pad_token_id] * padding_length
            padded_labels = example['labels'] + [-100] * padding_length # Pad labels with -100
            padded_attention_mask = example['attention_mask'] + [0] * padding_length
        else: # Should already be truncated by tokenizer(truncation=True)
            padded_input_ids = example['input_ids']
            padded_labels = example['labels']
            padded_attention_mask = example['attention_mask']

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long)
        }

def load_model_and_tokenizer(model_path, use_8bit=False, use_4bit=False, trust_remote_code=True):
    """Load model and tokenizer with specified quantization."""

    logger.info(f"🤖 Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Tokenizer pad_token set to eos_token: {tokenizer.pad_token}")

    model_kwargs = {
        'trust_remote_code': trust_remote_code,
        'device_map': 'auto',
        'torch_dtype': torch.float16
    }

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs['quantization_config'] = quantization_config
        logger.info("Using 4-bit quantization.")
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs['quantization_config'] = quantization_config
        logger.info("Using 8-bit quantization.")
    else:
        logger.info("Using full precision (float16) or bfloat16 if applicable.")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path} with specified kwargs. Attempting fallback: {e}", exc_info=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=trust_remote_code, torch_dtype=torch.float16)
            if torch.cuda.is_available():
                model.to('cuda')
                logger.info("Fallback: Loaded model to GPU manually.")
            else:
                logger.info("Fallback: Loaded model to CPU.")
        except Exception as fallback_e:
            logger.critical(f"Critical Error: Failed to load model even with fallback: {fallback_e}", exc_info=True)
            raise

    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)
        logger.info("Model prepared for k-bit training.")

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description='Mistral 7B LoRA Fine-tuning')

    parser.add_argument('--model-name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Name or path of the base model to fine-tune.')
    parser.add_argument('--data-file', type=str, default='dataset/private/tofu/tofu_train.json',
                        help='Path to the JSON file containing Q&A pairs.')
    parser.add_argument('--output-dir', type=str, default='./model_checkpoints/mistral_tofu_lora_tuned',
                        help='Directory to save the trained model and tokenizer.')

    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA attention dimension (r).')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='Alpha parameter for LoRA scaling.')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                        help='Dropout probability for LoRA layers.')

    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size per device during training.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Initial learning rate for AdamW optimizer.')
    parser.add_argument('--warmup-ratio', type=float, default=0.03,
                        help='Ratio of total training steps for warmup.')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay if we apply some.')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm (gradient clipping).')

    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length for tokenization.')
    parser.add_argument('--num-examples', type=int, default=None,
                        help='Number of examples to use from the dataset (None for all).')

    parser.add_argument('--use-4bit', action='store_true',
                        help='Enable 4-bit quantization (BitsAndBytes).')
    parser.add_argument('--use-8bit', action='store_true',
                        help='Enable 8-bit quantization (BitsAndBytes).')
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use FP16 (mixed precision) training.')
    parser.add_argument('--bf16', action='store_true',
                        help='Whether to use BF16 (bfloat16) training (preferred if GPU supports it).')

    parser.add_argument('--logging-steps', type=int, default=50,
                        help='Log every N updates steps.')
    parser.add_argument('--save-steps', type=int, default=500,
                        help='Save checkpoint every N updates steps.')
    parser.add_argument('--save-total-limit', type=int, default=1,
                        help='Limit the total amount of checkpoints saved.')

    args = parser.parse_args()

    if args.use_4bit and args.use_8bit:
        parser.error("Cannot use both --use-4bit and --use-8bit. Choose one or none.")
    if args.fp16 and args.bf16:
        parser.error("Cannot use both --fp16 and --bf16. Choose one or none.")
    if args.bf16 and not (torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8):
        logger.warning("BF16 is requested but GPU does not seem to support it (Ampere or newer required). Falling back to FP16 if specified, otherwise full precision.")
        if not args.fp16:
            args.fp16 = True


    logger.info("🚀 Starting Mistral 7B LoRA Fine-tuning")
    logger.info(f"⚙️ Config: LR={args.learning_rate}, Epochs={args.num_epochs}, Max_Grad_Norm={args.max_grad_norm}")
    logger.info(f"Quantization: 4-bit={args.use_4bit}, 8-bit={args.use_8bit}")
    logger.info(f"Precision: FP16={args.fp16}, BF16={args.bf16}")

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Dataset not found: {args.data_file}")

    with open(args.data_file, 'r') as f:
        data = json.load(f)

    if args.num_examples:
        data = data[:args.num_examples]
        logger.info(f"📊 Using {len(data)} examples (limited by --num-examples).")
    else:
        logger.info(f"📊 Using all {len(data)} examples.")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🎯 Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # The dataset now handles padding and tensor conversion directly
    train_dataset = MistralInstructionDataset(data, tokenizer, args.max_length)
    # DataCollatorForLanguageModeling is still used for its label shifting logic,
    # but padding will be minimal as data is pre-padded.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,

        fp16=args.fp16,
        bf16=args.bf16,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        evaluation_strategy="no",
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        remove_unused_columns=False, # IMPORTANT: Keep this False when custom handling data like this
        dataloader_num_workers=os.cpu_count() // 2 or 1,
        report_to=None,

        save_total_limit=args.save_total_limit,
        seed=42,
        data_seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("🚀 Starting training...")
    try:
        trainer.train()
        logger.info("✅ Training completed!")

        logger.info(f"💾 Saving final model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        logger.info("🎉 Done! Model and tokenizer saved successfully.")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        if trainer.state.global_step > 0:
            logger.info(f"Attempting to save last checkpoint after error at step {trainer.state.global_step}...")
            trainer.save_model(os.path.join(args.output_dir, f"checkpoint-last-error-step-{trainer.state.global_step}"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, f"checkpoint-last-error-step-{trainer.state.global_step}"))


if __name__ == "__main__":
    main()