"""
Model optimization utilities for HuggingFace models.
Provides functions for model quantization, flash attention, and other optimizations.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from typing import Tuple, Any, Dict, Optional
from pathlib import Path

from core.settings import settings
from core.exceptions import ModelOptimizationError

logger = logging.getLogger(__name__)

def get_optimized_model(
    model_name: str,
    use_4bit: bool = True,
    use_8bit: bool = False,
    use_flash_attention: bool = True,
    use_lora: bool = True,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05
) -> tuple[Any, Any]:
    """
    Get an optimized model and tokenizer for training.
    
    Args:
        model_name: Name or path of the model
        use_4bit: Whether to use 4-bit quantization
        use_8bit: Whether to use 8-bit quantization
        use_flash_attention: Whether to use flash attention
        use_lora: Whether to use LoRA
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=use_flash_attention
        )
        
        # Prepare model for k-bit training if using quantization
        if use_4bit or use_8bit:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA if enabled
        if use_lora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error optimizing model: {str(e)}")
        raise ModelOptimizationError(f"Failed to optimize model: {str(e)}")

def get_optimized_training_args(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_grad_norm: float = 0.3,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    evaluation_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    fp16: bool = True,
    report_to: list = None
) -> TrainingArguments:
    """
    Get optimized training arguments.
    
    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm
        warmup_ratio: Ratio of warmup steps
        lr_scheduler_type: Learning rate scheduler type
        logging_steps: Number of steps between logging
        save_strategy: Model saving strategy
        evaluation_strategy: Evaluation strategy
        load_best_model_at_end: Whether to load best model at end
        fp16: Whether to use mixed precision training
        report_to: List of integrations to report to
        
    Returns:
        TrainingArguments object
    """
    try:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=load_best_model_at_end,
            fp16=fp16,
            report_to=report_to or ["tensorboard"],
            dataloader_num_workers=settings.get("optimization.dataloader_num_workers", 4),
            dataloader_pin_memory=settings.get("optimization.dataloader_pin_memory", True)
        )
    except Exception as e:
        logger.error(f"Error creating training arguments: {str(e)}")
        raise ModelOptimizationError(f"Failed to create training arguments: {str(e)}") 