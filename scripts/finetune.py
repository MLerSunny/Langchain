#!/usr/bin/env python3
"""
Fine-tuning script for DeepSeek models.

This script implements fine-tuning of DeepSeek models using low-rank adaptation (LoRA)
and QLoRA techniques.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from transformers import Trainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import (
    LEARNING_RATE,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS,
    MAX_STEPS,
    LOGGING_STEPS,
    SAVE_STEPS,
    WARMUP_STEPS,
    OUTPUT_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model we are fine-tuning."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank parameter for LoRA"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Alpha parameter for LoRA"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit quantization (fp4 or nf4)"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are training on."""

    dataset_path: str = field(
        default=TRAINING_DATASET_PATH,
        metadata={"help": "Path to the training dataset"}
    )
    eval_dataset_path: str = field(
        default=EVAL_DATASET_PATH,
        metadata={"help": "Path to the evaluation dataset"}
    )
    max_seq_length: int = field(
        default=CONTEXT_WINDOW,
        metadata={"help": "Maximum sequence length to use for training"}
    )


def load_and_prepare_datasets(
    tokenizer,
    data_args: DataArguments,
) -> Union[Dict[str, Dataset], None]:
    """
    Load and prepare datasets for training and evaluation.

    Args:
        tokenizer: Tokenizer to use for tokenization.
        data_args: Data arguments.

    Returns:
        Dictionary containing datasets, or None if dataset not found.
    """
    # Check if dataset files exist
    if not os.path.exists(data_args.dataset_path):
        logger.error(f"Training dataset not found at {data_args.dataset_path}")
        return None

    # Load training dataset
    logger.info(f"Loading training dataset from {data_args.dataset_path}")
    
    # Determine the format based on file extension
    if data_args.dataset_path.endswith(".json") or data_args.dataset_path.endswith(".jsonl"):
        # Load JSON dataset
        train_dataset = load_dataset("json", data_files=data_args.dataset_path, split="train")
    else:
        # Assume directory structure readable by load_dataset
        train_dataset = load_dataset(data_args.dataset_path, split="train")
    
    # Load evaluation dataset if available
    eval_dataset = None
    if os.path.exists(data_args.eval_dataset_path):
        logger.info(f"Loading evaluation dataset from {data_args.eval_dataset_path}")
        if data_args.eval_dataset_path.endswith(".json") or data_args.eval_dataset_path.endswith(".jsonl"):
            eval_dataset = load_dataset("json", data_files=data_args.eval_dataset_path, split="train")
        else:
            eval_dataset = load_dataset(data_args.eval_dataset_path, split="train")
    
    # Define tokenization function
    def tokenize_function(examples):
        # Assume dataset has 'prompt' and 'completion' fields
        # For DeepSeek, we need to format them correctly
        input_texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Format for DeepSeek Coder
            text = f"{prompt}{completion}"
            input_texts.append(text)
        
        tokenized = tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
            return_tensors="pt",
        )
        
        # Set labels equal to input_ids for causal LM training
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Apply tokenization
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    if eval_dataset:
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
    else:
        tokenized_eval = None
    
    return {
        "train": tokenized_train,
        "eval": tokenized_eval,
    }


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> None:
    """
    Fine-tune the model.

    Args:
        model_args: Model arguments.
        data_args: Data arguments.
        training_args: Training arguments.
    """
    # Set device map
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
    # Setup quantization config if using 4-bit
    compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
    quantization_config = None
    
    if model_args.use_4bit:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )
    elif model_args.use_8bit:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    
    # Prepare model for kbit training if using quantization
    if model_args.use_4bit or model_args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA if requested
    if model_args.use_lora:
        logger.info("Using LoRA for training")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load datasets
    datasets = load_and_prepare_datasets(tokenizer, data_args)
    if not datasets:
        logger.error("Failed to load datasets. Exiting.")
        return
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    

def main():
    """Parse arguments and start training."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # Parse args from command line or from a file
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # Add default training arguments
        sys.argv.extend([
            "--output_dir", OUTPUT_DIR,
            "--learning_rate", str(LEARNING_RATE),
            "--per_device_train_batch_size", str(BATCH_SIZE),
            "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
            "--max_steps", str(MAX_STEPS),
            "--logging_steps", str(LOGGING_STEPS),
            "--save_steps", str(SAVE_STEPS),
            "--warmup_steps", str(WARMUP_STEPS),
            "--num_train_epochs", str(NUM_EPOCHS),
            "--lr_scheduler_type", "cosine",
            "--report_to", "tensorboard",
            "--overwrite_output_dir",
        ])
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Call train function
    train(model_args, data_args, training_args)


if __name__ == "__main__":
    main() 