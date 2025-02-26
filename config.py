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

def validate_config():
    """Validate that all required environment variables are set"""
    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN environment variable is not set. Please check your .env file.")
    
    if not HF_MODEL_NAME:
        raise ValueError("HF_MODEL_NAME environment variable is not set. Please check your .env file.") 