#!/bin/bash
# Bash script for fine-tuning DeepSeek-LLM
# Usage: ./scripts/finetune.sh

# Configuration
model_name=""
dataset_dir="data/training"
config_file="core/deepspeed_zero3.json"

# Default values
output_dir="checkpoints/r1_ins_lora"
train_epochs=3
batch_size=1
grad_accum_steps=8

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config-file)
      config_file="$2"
      shift
      ;;
    --output-dir)
      output_dir="$2"
      shift
      ;;
    --train-epochs)
      train_epochs="$2"
      shift
      ;;
    --batch-size)
      batch_size="$2"
      shift
      ;;
    --grad-accum-steps)
      grad_accum_steps="$2"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

echo "Starting DeepSeek-LLM fine-tuning with QLoRA..."

# Ensure the config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: DeepSpeed config file $config_file does not exist!"
    exit 1
fi

# Ensure the dataset directory exists
if [ ! -d "finetune/dataset" ]; then
    echo "Error: Dataset directory 'finetune/dataset' does not exist or is empty!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"
echo "Created output directory: $output_dir"

# Set environment variables for training
export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="false"

# Build the DeepSpeed command
deepspeed_command="deepspeed --num_gpus=1 finetune/trainer.py \
                    --deepspeed $config_file \
                    --model_name_or_path deepseek-ai/deepseek-llm-32b-instruct \
                    --dataset_dir finetune/dataset \
                    --output_dir $output_dir \
                    --num_train_epochs $train_epochs \
                    --per_device_train_batch_size $batch_size \
                    --per_device_eval_batch_size $batch_size \
                    --gradient_accumulation_steps $grad_accum_steps \
                    --learning_rate 2e-4 \
                    --warmup_ratio 0.03 \
                    --lr_scheduler_type 'cosine' \
                    --logging_steps 10 \
                    --save_steps 100 \
                    --eval_steps 100 \
                    --save_total_limit 3 \
                    --max_seq_length 2048 \
                    --load_in_4bit \
                    --lora_r 64 \
                    --lora_alpha 128 \
                    --lora_dropout 0.05"

# Run the fine-tuning command
echo "Running: $deepspeed_command"
eval $deepspeed_command

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Fine-tuning completed successfully!"
    echo "Model checkpoint saved to: $output_dir"
else
    echo "Fine-tuning failed with exit code $?"
    exit 1
fi 