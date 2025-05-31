import os
import yaml
import requests
import json
import gradio as gr
import logging
from typing import Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.base_url = f"http://{config['host']}:{config['port']}/api"
    
    def check_server_status(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = self.session.get(f"{self.base_url}/tags")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None) -> str:
        """Generate response from Ollama"""
        if not self.check_server_status():
            raise ConnectionError("Ollama server is not running")
        
        max_tokens = max_tokens or self.config["default_max_tokens"]
        temperature = temperature or self.config["default_temperature"]
        
        data = {
            "model": self.config["model_name"],
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": self.config["stream"]
        }
        
        try:
            response = self.session.post(f"{self.base_url}/generate", json=data)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

def load_config():
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "model_training.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def generate_response(prompt: str, max_tokens: Optional[int] = None, 
                     temperature: Optional[float] = None) -> str:
    """
    Generate a response using the Ollama API with retry mechanism
    """
    config = load_config()
    client = OllamaClient(config["inference"]["ollama"])
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            return client.generate(prompt, max_tokens, temperature)
        except ConnectionError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Connection error, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return f"Error: Could not connect to Ollama server. Please ensure it's running."
        except Exception as e:
            return f"Error: {str(e)}"

def create_gradio_interface():
    """
    Create a Gradio interface for the model with enhanced features
    """
    config = load_config()
    gradio_config = config["inference"]["gradio"]
    
    def process_with_progress(prompt: str, max_tokens: int, temperature: float):
        """Process the prompt with progress feedback"""
        try:
            with gr.Progress() as progress:
                progress(0, desc="Generating response...")
                response = generate_response(prompt, max_tokens, temperature)
                progress(1.0, desc="Complete!")
                return response
        except Exception as e:
            logger.error(f"Error in Gradio interface: {str(e)}")
            return f"Error: {str(e)}"
    
    iface = gr.Interface(
        fn=process_with_progress,
        inputs=[
            gr.Textbox(
                label="Input Prompt",
                lines=gradio_config["input_lines"],
                placeholder="Enter your prompt here..."
            ),
            gr.Slider(
                minimum=gradio_config["max_tokens_range"][0],
                maximum=gradio_config["max_tokens_range"][1],
                value=config["inference"]["ollama"]["default_max_tokens"],
                label="Max Tokens",
                step=1
            ),
            gr.Slider(
                minimum=gradio_config["temperature_range"][0],
                maximum=gradio_config["temperature_range"][1],
                value=config["inference"]["ollama"]["default_temperature"],
                label="Temperature",
                step=0.1
            )
        ],
        outputs=[
            gr.Textbox(
                label="Generated Response",
                lines=gradio_config["output_lines"],
                show_copy_button=True
            )
        ],
        title=f"{config['model']['name']} Chat Interface",
        description="Enter your prompt and adjust the generation parameters.",
        examples=[
            ["What is machine learning?", 512, 0.7],
            ["Explain quantum computing", 1024, 0.8],
            ["Write a short poem about AI", 256, 0.9]
        ],
        cache_examples=True
    )
    return iface

if __name__ == "__main__":
    try:
        # Create and launch the Gradio interface
        interface = create_gradio_interface()
        interface.launch(
            share=load_config()["inference"]["gradio"]["share"],
            server_name="0.0.0.0",
            server_port=7860
        )
    except Exception as e:
        logger.error(f"Failed to launch interface: {str(e)}")
        raise 