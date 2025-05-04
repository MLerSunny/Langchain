#!/bin/bash
# Bash script for fine-tuning DeepSeek-LLM
# Usage: ./scripts/finetune.sh

# Default configuration
model_name="deepseek-llm:7b"
dataset_path="finetune/dataset/insurance_conversations.json"
output_dir="data/models"
learning_rate="2e-5"
batch_size=1
gradient_accumulation_steps=4
num_epochs=3
max_steps=-1
logging_steps=10
save_steps=100
warmup_steps=50
lora_r=64
lora_alpha=128
lora_dropout=0.05
use_lora=true
use_4bit=true
max_seq_length=2048

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_name)
      model_name="$2"
      shift 2
      ;;
    --dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    --output_dir)
      output_dir="$2"
      shift 2
      ;;
    --learning_rate)
      learning_rate="$2"
      shift 2
      ;;
    --batch_size)
      batch_size="$2"
      shift 2
      ;;
    --gradient_accumulation_steps)
      gradient_accumulation_steps="$2"
      shift 2
      ;;
    --num_epochs)
      num_epochs="$2"
      shift 2
      ;;
    --max_steps)
      max_steps="$2"
      shift 2
      ;;
    --logging_steps)
      logging_steps="$2"
      shift 2
      ;;
    --save_steps)
      save_steps="$2"
      shift 2
      ;;
    --warmup_steps)
      warmup_steps="$2"
      shift 2
      ;;
    --lora_r)
      lora_r="$2"
      shift 2
      ;;
    --lora_alpha)
      lora_alpha="$2"
      shift 2
      ;;
    --lora_dropout)
      lora_dropout="$2"
      shift 2
      ;;
    --use_lora)
      use_lora="$2"
      shift 2
      ;;
    --use_4bit)
      use_4bit="$2"
      shift 2
      ;;
    --max_seq_length)
      max_seq_length="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

echo "Starting DeepSeek-LLM fine-tuning..."
echo "Model: $model_name"
echo "Dataset: $dataset_path"
echo "Output directory: $output_dir"

# Ensure the dataset file exists
if [ ! -f "$dataset_path" ]; then
    echo "Error: Dataset file $dataset_path does not exist!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"
echo "Created output directory: $output_dir"

# Set environment variables for training
export TOKENIZERS_PARALLELISM="false"

# Determine if we should use GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, using CUDA"
    export CUDA_VISIBLE_DEVICES="0"
    gpu_available=true
else
    echo "No NVIDIA GPU detected, using CPU only (this will be slow)"
    gpu_available=false
    use_4bit=false
fi

# Convert boolean params to Python booleans
if [ "$use_lora" = "true" ]; then
    use_lora_param="True"
else
    use_lora_param="False"
fi

if [ "$use_4bit" = "true" ]; then
    use_4bit_param="True"
else
    use_4bit_param="False"
fi

# Build the command
command="python scripts/finetune.py \
    --model_name_or_path $model_name \
    --dataset_path $dataset_path \
    --output_dir $output_dir \
    --learning_rate $learning_rate \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs $num_epochs \
    --max_steps $max_steps \
    --logging_steps $logging_steps \
    --save_steps $save_steps \
    --warmup_steps $warmup_steps \
    --use_lora $use_lora_param \
    --lora_rank $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --use_4bit $use_4bit_param \
    --max_seq_length $max_seq_length \
    --report_to tensorboard"

# Run the fine-tuning command
echo "Running: $command"
eval $command

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Fine-tuning completed successfully!"
    echo "Model checkpoint saved to: $output_dir"
else
    echo "Fine-tuning failed with exit code $?"
    exit 1
fi 