# Product Recommendation System

This project implements a product recommendation system using session data and embeddings.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Copy the template file
cp .env.template .env

# Edit .env with your actual values
# Get your Hugging Face API token from: https://huggingface.co/settings/tokens
```

3. Update the `.env` file with your actual values:
- `HF_API_TOKEN`: Your Hugging Face API token
- `HF_MODEL_NAME`: The model to use for embeddings (default: intfloat/multilingual-e5-large)
- `TEST_MODE`: Set to "true" for testing with limited data

## Security Notes

- Never commit the `.env` file containing actual secrets
- Use `.env.template` as a template for required environment variables
- The `.gitignore` file is configured to prevent committing sensitive files

## Usage

1. Process product data:
```bash
python preprocess_products.py
```

2. Generate session graph:
```bash
python node_gen.py
```

3. Test embeddings:
```bash
python testhf.py
```
