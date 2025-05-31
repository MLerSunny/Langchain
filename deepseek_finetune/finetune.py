import os
import torch
import yaml
import logging
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset
import wandb
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "model_training.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def validate_model_and_tokenizer(model, tokenizer):
    """Validate model and tokenizer setup"""
    try:
        # Test tokenizer
        test_text = "Hello, this is a test."
        tokens = tokenizer(test_text, return_tensors="pt")
        logger.info("Tokenizer validation successful")
        
        # Test model forward pass
        with torch.no_grad():
            outputs = model(**tokens)
        logger.info("Model validation successful")
        return True
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

def setup_model_and_tokenizer(config):
    try:
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
        tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Initializing model...")
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": getattr(torch, config["fine_tuning"]["quantization"]["dtype"])
        }
        
        if config["fine_tuning"]["quantization"]["enabled"]:
            model_kwargs["load_in_4bit"] = True
        
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["base_model"],
            **model_kwargs
        )
        
        if config["fine_tuning"]["quantization"]["enabled"]:
            logger.info("Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(model)
        
        if config["fine_tuning"]["lora"]["enabled"]:
            logger.info("Configuring LoRA...")
            lora_config = LoraConfig(
                r=config["fine_tuning"]["lora"]["r"],
                lora_alpha=config["fine_tuning"]["lora"]["alpha"],
                target_modules=config["fine_tuning"]["lora"]["target_modules"],
                lora_dropout=config["fine_tuning"]["lora"]["dropout"],
                bias=config["fine_tuning"]["lora"]["bias"],
                task_type=config["fine_tuning"]["lora"]["task_type"]
            )
            model = get_peft_model(model, lora_config)
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to setup model and tokenizer: {str(e)}")
        raise

def prepare_dataset(tokenizer, config):
    try:
        logger.info("Loading dataset...")
        dataset_path = os.path.join(config["paths"]["dataset"], "training_data.json")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        
        dataset = load_dataset("json", data_files=dataset_path)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=config["fine_tuning"]["dataset"]["truncation"],
                max_length=config["fine_tuning"]["dataset"]["max_length"],
                padding=config["fine_tuning"]["dataset"]["padding"]
            )
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
        raise

def main():
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Create necessary directories
        Path(config["paths"]["checkpoints"]).mkdir(parents=True, exist_ok=True)
        Path(config["paths"]["final_model"]).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        wandb.init(project="deepseek-finetune")
        logger.info("Wandb initialized")
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Validate model and tokenizer
        if not validate_model_and_tokenizer(model, tokenizer):
            raise RuntimeError("Model validation failed")
        
        # Prepare dataset
        dataset = prepare_dataset(tokenizer, config)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config["paths"]["checkpoints"],
            num_train_epochs=config["fine_tuning"]["training"]["epochs"],
            per_device_train_batch_size=config["fine_tuning"]["training"]["batch_size"],
            gradient_accumulation_steps=config["fine_tuning"]["training"]["gradient_accumulation_steps"],
            learning_rate=config["fine_tuning"]["training"]["learning_rate"],
            fp16=config["fine_tuning"]["training"]["fp16"],
            logging_steps=config["fine_tuning"]["training"]["logging_steps"],
            save_strategy=config["fine_tuning"]["training"]["save_strategy"],
            evaluation_strategy=config["fine_tuning"]["training"]["evaluation_strategy"],
            load_best_model_at_end=config["fine_tuning"]["training"]["load_best_model_at_end"],
            report_to="wandb",
            # Additional arguments for better training
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            save_total_limit=3,
            metric_for_best_model="eval_loss"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        
        logger.info("Starting training...")
        # Start training
        trainer.train()
        
        logger.info("Training completed. Saving model...")
        # Save the model
        trainer.save_model(config["paths"]["final_model"])
        tokenizer.save_pretrained(config["paths"]["final_model"])
        
        logger.info("Model saved successfully")
        wandb.finish()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 