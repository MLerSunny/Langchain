#!/bin/bash
# Bash script for serving the RAG system with FastAPI
# Usage: ./scripts/serve.sh [--model model_name_or_path]

# Default values
model="deepseek-ai/deepseek-llm-32b-instruct"
use_lora=false
lora_path="checkpoints/r1_ins_lora/final"
port=8080

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      model="$2"
      shift
      ;;
    --use-lora)
      use_lora=true
      ;;
    --lora-path)
      lora_path="$2"
      shift
      ;;
    --port)
      port="$2"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

# Check if lora path exists when use_lora is enabled
if [ "$use_lora" = true ] && [ ! -d "$lora_path" ]; then
    echo "Error: LoRA adapter path '$lora_path' does not exist!"
    exit 1
fi

# Start vLLM server in a separate process
vllm_port=8000
vllm_command="python -m vllm.entrypoints.openai.api_server --model $model --port $vllm_port --tensor-parallel-size 1"

# Add LoRA adapter if enabled
if [ "$use_lora" = true ]; then
    vllm_command+=" --peft-model $lora_path"
fi

# Start vLLM server in the background
echo "Starting vLLM server..."
$vllm_command > vllm.log 2>&1 &
vllm_pid=$!
echo "Started vLLM server with PID: $vllm_pid"

# Wait for vLLM server to start up
echo "Waiting for vLLM server to start..."
sleep 10

# Set environment variables
export VLLM_HOST="http://localhost:$vllm_port"
export CHROMA_DIR="chroma_insurance"
export COLLECTION_NAME="insurance_docs"

# Start the FastAPI server
echo "Starting FastAPI server on port $port..."
api_command="uvicorn serve.main:app --host 0.0.0.0 --port $port --reload"

echo "Running: $api_command"
echo "API server is running. Press Ctrl+C to stop."

# Handle cleanup when the script is interrupted
cleanup() {
    echo "Stopping vLLM server (PID: $vllm_pid)..."
    kill $vllm_pid
    exit 0
}

# Register the cleanup function for these signals
trap cleanup SIGINT SIGTERM

# Run the API server
exec $api_command 