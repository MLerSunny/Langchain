# Deepseek-LLM-7B Fine-tuning and Inference

This project provides scripts for fine-tuning the Deepseek-LLM-7B model using Hugging Face's transformers library and running inference using Ollama.

## Prerequisites

1. Python 3.8 or higher
2. CUDA-capable GPU (recommended for fine-tuning)
3. Ollama installed and running locally
4. Hugging Face account and access token

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token:
```bash
huggingface-cli login
```

4. Pull the Deepseek-LLM-7B model in Ollama:
```bash
ollama pull deepseek-llm-7b
```

## Fine-tuning

1. Prepare your dataset in JSON format with a "text" field containing the training data.

2. Update the dataset path in `finetune.py`:
```python
dataset = prepare_dataset(tokenizer, "path_to_your_dataset.json")
```

3. Run the fine-tuning script:
```bash
python finetune.py
```

The script uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and includes:
- 4-bit quantization
- Gradient checkpointing
- Mixed precision training
- Wandb integration for monitoring

## Inference

1. Make sure Ollama is running:
```bash
ollama serve
```

2. Run the inference script:
```bash
python inference.py
```

This will launch a Gradio web interface where you can:
- Enter prompts
- Adjust generation parameters (max tokens, temperature)
- Get model responses

## Model Parameters

The fine-tuning script uses the following default parameters:
- LoRA rank (r): 16
- LoRA alpha: 32
- Learning rate: 2e-4
- Batch size: 4
- Gradient accumulation steps: 4
- Training epochs: 3

You can modify these parameters in the `finetune.py` script.

## Notes

- The fine-tuned model will be saved in the `./final_model` directory
- Training checkpoints are saved in the `./checkpoints` directory
- Make sure you have enough GPU memory for fine-tuning (at least 16GB recommended)
- The inference script assumes Ollama is running on the default port (11434) 