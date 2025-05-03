"""
QLoRA fine-tuning script for DeepSeek-LLM models.
This script loads a ShareGPT-style dataset and fine-tunes the model using QLoRA.
"""

import os
import json
import argparse
import sys
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from datasets import load_dataset
from rouge_score import rouge_scorer

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name: str = field(
        default=settings.model_name,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"},
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention for faster training"},
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""
    dataset_dir: str = field(
        default="data/training",
        metadata={"help": "Directory containing JSON datasets in ShareGPT format"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for training"},
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension"},
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "Alpha parameter for LoRA scaling"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"},
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of module names to apply LoRA to"},
    )


@dataclass
class EvalArguments:
    """Arguments for evaluation configuration."""
    eval_dataset_path: str = field(
        default="data/evaluation",
        metadata={"help": "Path to evaluation dataset for RAG callback"},
    )
    eval_every_n_steps: int = field(
        default=100,
        metadata={"help": "How often to run the RAG evaluation callback"},
    )


def format_sharegpt_dataset(examples):
    """
    Format ShareGPT style dataset for instruction fine-tuning.
    
    Args:
        examples: Raw examples from dataset
        
    Returns:
        Formatted examples with prompt and completion
    """
    formatted_examples = []
    
    for example in examples:
        conversations = example.get("conversations", [])
        if not conversations or len(conversations) < 2:
            continue
        
        # Extract system prompt if it exists
        system_prompt = ""
        start_idx = 0
        if conversations[0]["from"].lower() == "system":
            system_prompt = conversations[0]["value"]
            start_idx = 1
        
        # Process conversation pairs (human -> assistant)
        for i in range(start_idx, len(conversations) - 1, 2):
            if i + 1 < len(conversations):
                if conversations[i]["from"].lower() in ["human", "user"] and \
                   conversations[i+1]["from"].lower() in ["assistant", "gpt"]:
                    
                    user_prompt = conversations[i]["value"]
                    assistant_response = conversations[i+1]["value"]
                    
                    # Format with system prompt if available
                    if system_prompt:
                        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
                    else:
                        prompt = f"<|user|>\n{user_prompt}\n<|assistant|>\n"
                    
                    formatted_examples.append({
                        "prompt": prompt,
                        "completion": assistant_response
                    })
    
    return formatted_examples


def prepare_dataset(data_args):
    """
    Prepare dataset for training from ShareGPT JSON files.
    
    Args:
        data_args: Data arguments
        
    Returns:
        Processed dataset ready for training
    """
    # Load all JSON files from the dataset directory
    json_files = [f for f in os.listdir(data_args.dataset_dir) if f.endswith('.json')]
    
    if not json_files:
        raise ValueError(f"No JSON files found in {data_args.dataset_dir}")
    
    all_examples = []
    for json_file in json_files:
        file_path = os.path.join(data_args.dataset_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
                if isinstance(examples, dict):
                    examples = [examples]  # Convert single example to list
                all_examples.extend(examples)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    # Format dataset
    formatted_examples = format_sharegpt_dataset(all_examples)
    
    # Create training splits (90% train, 10% eval)
    train_size = int(0.9 * len(formatted_examples))
    train_examples = formatted_examples[:train_size]
    eval_examples = formatted_examples[train_size:]
    
    logger.info(f"Loaded {len(formatted_examples)} examples, {len(train_examples)} for training, {len(eval_examples)} for evaluation")
    
    # Create dataset objects
    dataset = {
        "train": train_examples,
        "validation": eval_examples
    }
    
    return dataset


def prepare_deepseek_for_training(model_args, lora_args):
    """
    Prepare DeepSeek model for QLoRA fine-tuning.
    
    Args:
        model_args: Model arguments
        lora_args: LoRA configuration arguments
        
    Returns:
        Prepared model and tokenizer
    """
    # Check if CUDA is available
    use_quantization = torch.cuda.is_available()
    
    # Configure quantization if available
    if use_quantization:
        # Configure quantization
        logger.info("Using 4-bit quantization for training")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with quantization config
        logger.info(f"Loading model: {model_args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name,
            quantization_config=bnb_config,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch.float16,
            use_cache=False,  # Disable KV cache during training
            device_map="auto",
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        # Load model without quantization for Windows compatibility
        logger.info(f"Loading model without quantization: {model_args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch.float32,  # Use float32 for CPU training
            use_cache=False,  # Disable KV cache during training
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print model info
    logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    return model, tokenizer


class RAGEvaluationCallback(TrainerCallback):
    """
    Custom callback to evaluate model performance on RAG tasks during training.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        eval_dataset_path: str, 
        every_n_steps: int = 100,
        max_eval_samples: int = 5,
    ):
        """
        Initialize the RAG evaluation callback.
        
        Args:
            model: The model being trained
            tokenizer: Tokenizer for the model
            eval_dataset_path: Path to evaluation dataset
            every_n_steps: Run evaluation every N steps
            max_eval_samples: Maximum number of samples to evaluate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset_path = eval_dataset_path
        self.every_n_steps = every_n_steps
        self.max_eval_samples = max_eval_samples
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.eval_samples = self._load_eval_samples()
        
    def _load_eval_samples(self) -> List[Dict]:
        """
        Load evaluation samples from the dataset path.
        
        Returns:
            List of evaluation samples with queries and reference answers
        """
        eval_samples = []
        
        # Check if evaluation directory exists
        if not os.path.exists(self.eval_dataset_path):
            logger.warning(f"Evaluation dataset path {self.eval_dataset_path} does not exist")
            return eval_samples
        
        # Load all JSON files from the evaluation directory
        json_files = [f for f in os.listdir(self.eval_dataset_path) if f.endswith('.json')]
        if not json_files:
            logger.warning(f"No JSON files found in {self.eval_dataset_path}")
            return eval_samples
        
        # Load samples from JSON files
        for json_file in json_files:
            file_path = os.path.join(self.eval_dataset_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        eval_samples.extend(data)
                    elif isinstance(data, dict):
                        eval_samples.append(data)
            except Exception as e:
                logger.error(f"Error loading evaluation sample from {file_path}: {e}")
        
        # Ensure we have query/answer pairs
        eval_samples = [
            sample for sample in eval_samples 
            if "query" in sample and "reference_answer" in sample
        ]
        
        # Limit number of samples
        if len(eval_samples) > self.max_eval_samples:
            eval_samples = random.sample(eval_samples, self.max_eval_samples)
            
        logger.info(f"Loaded {len(eval_samples)} evaluation samples for RAG callback")
        return eval_samples
    
    def _generate_answer(self, query: str, context: str = "") -> str:
        """
        Generate an answer using the current model state.
        
        Args:
            query: The query to answer
            context: Optional context for RAG
            
        Returns:
            Generated answer
        """
        try:
            # Prepare the prompt with context if available
            if context:
                prompt = f"""You are a helpful AI assistant. Use the following information to answer the question. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.

                Context:
                {context}

                Question: {query}

                Answer:"""
            else:
                prompt = f"""You are a helpful AI assistant. Answer the following question.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.

                Question: {query}

                Answer:"""
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate the answer
            with torch.no_grad():
                output = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the answer part (after the prompt)
            answer = generated_text[len(prompt):].strip()
            return answer
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "[Error generating answer]"
    
    def _evaluate_samples(self) -> Dict[str, float]:
        """
        Evaluate the model on all evaluation samples.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.eval_samples:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for sample in self.eval_samples:
            query = sample["query"]
            reference_answer = sample["reference_answer"]
            context = sample.get("context", "")
            
            # Generate answer
            generated_answer = self._generate_answer(query, context)
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
            
            rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
            rouge2_scores.append(rouge_scores["rouge2"].fmeasure)
            rougeL_scores.append(rouge_scores["rougeL"].fmeasure)
        
        # Calculate average scores
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
        
        return {
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL
        }
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each step during training.
        
        Args:
            args: Training arguments
            state: Training state
            control: Control object
        """
        # Only evaluate every N steps and on the first step
        if state.global_step % self.every_n_steps == 0 or state.global_step == 1:
            logger.info(f"Running RAG evaluation at step {state.global_step}")
            
            # Evaluate samples
            metrics = self._evaluate_samples()
            
            # Log metrics
            for metric_name, value in metrics.items():
                logger.info(f"RAG Eval {metric_name}: {value:.4f}")
                
                # Log to tensorboard if enabled
                if args.report_to and "tensorboard" in args.report_to:
                    for callback in kwargs.get("callbacks", []):
                        if hasattr(callback, "writer") and callback.writer:
                            callback.writer.add_scalar(
                                f"rag_eval/{metric_name}", 
                                value, 
                                state.global_step
                            )


def train(model_args, data_args, lora_args, training_args, eval_args):
    """
    Train the model with the given arguments.
    
    Args:
        model_args: Model configuration arguments
        data_args: Dataset configuration arguments
        lora_args: LoRA configuration arguments
        training_args: Training configuration arguments
        eval_args: Evaluation configuration arguments
    """
    # Prepare the model and tokenizer
    model, tokenizer = prepare_deepseek_for_training(model_args, lora_args)
    
    # Prepare the dataset
    dataset = prepare_dataset(data_args)
    
    # Data collator
    def data_collator(examples):
        # Tokenize inputs
        batch = tokenizer(
            [x["prompt"] + x["completion"] for x in examples],
            padding="max_length",
            max_length=data_args.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Create labels (shift inputs for causal LM)
        batch["labels"] = batch["input_ids"].clone()
        
        # Mask prompt tokens (we only want to compute loss on the completion)
        for i, example in enumerate(examples):
            prompt_len = len(tokenizer(example["prompt"], return_tensors="pt")["input_ids"][0])
            batch["labels"][i, :prompt_len] = -100  # Don't compute loss on prompt
            
        return batch
    
    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,  # Already a TrainingArguments object
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()


def main():
    """
    Parse command line arguments and start training.
    """
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek LLM with QLoRA")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint saving frequency")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Learning rate warmup steps")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--eval_dataset_path", type=str, default="", help="Path to evaluation dataset")
    
    args = parser.parse_args()
    
    # Create model arguments
    model_args = ModelArguments(
        model_name=args.model_name,
    )
    
    # Create data arguments
    data_args = DataArguments(
        dataset_dir=args.dataset_dir,
        max_seq_length=args.max_seq_length,
    )
    
    # Create LoRA arguments
    lora_args = LoraArguments(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Create training arguments
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=float(args.num_epochs),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=float(args.learning_rate),
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        fp16=torch.cuda.is_available(),  # Only use fp16 when CUDA is available
        optim="adamw_torch",
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # Create evaluation arguments
    eval_args = EvalArguments(
        eval_dataset_path=args.eval_dataset_path,
    )
    
    # Start training
    train(model_args, data_args, lora_args, training_args, eval_args)


if __name__ == "__main__":
    main() 