from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
env_path = Path('.env')
load_dotenv(dotenv_path=env_path)

# Hugging Face configuration
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
HF_MODEL_NAME = os.getenv('HF_MODEL_NAME', 'intfloat/multilingual-e5-large')

# Test mode configuration
TEST_MODE = os.getenv('TEST_MODE', 'false').lower() == 'true'

# GPU Configuration
USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'
GPU_ID = int(os.getenv('GPU_ID', '0'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '4'))

def validate_config():
    """Validate that all required environment variables are set"""
    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN environment variable is not set. Please check your .env file.")
    
    if not HF_MODEL_NAME:
        raise ValueError("HF_MODEL_NAME environment variable is not set. Please check your .env file.")
    
    if GPU_ID >= 0 and USE_GPU:
        try:
            import torch
            if not torch.cuda.is_available():
                print("Warning: GPU requested but CUDA is not available. Falling back to CPU.")
        except ImportError:
            print("Warning: PyTorch not installed. GPU acceleration will not be available.") 