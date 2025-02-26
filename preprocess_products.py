import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from llm_utils import LLMHandler
from config import HF_API_TOKEN, validate_config

def normalize_price(price):
    """Normalize price to range 0-1000"""
    try:
        return float(price)
    except (ValueError, TypeError):
        return None

def combine_brand_model(row):
    """Combine brand and model into single string"""
    brand = str(row['brand']) if pd.notna(row['brand']) else ''
    model = str(row['model']) if pd.notna(row['model']) else ''
    return f"Brand: {brand} , Model: {model}".strip()

def preprocess_products(chunk, llm_handler: LLMHandler):
    """Preprocess a chunk of product data"""
    # Make a copy to avoid modifying original data
    chunk = chunk.copy()
    
    # Convert price to numeric and normalize
    chunk['price'] = chunk['price'].apply(normalize_price)
    
    # Remove any invalid prices
    chunk = chunk.dropna(subset=['price'])
    
    # Normalize prices to 0-1000 range
    min_price = 0
    max_price = 99999
    chunk['normalized_price'] = ((chunk['price'] - min_price) / (max_price - min_price)) * 1000
    
    # Combine brand and model
    chunk['brand_model'] = chunk.apply(combine_brand_model, axis=1)
    
    # Generate embeddings for different fields
    fields_to_embed = {
        'title': chunk['title'].tolist()[:500],  # Testing with smaller subset
        'desc': chunk['desc'].tolist()[:500],
        'brand_model': chunk['brand_model'].tolist()[:500],
        'size': chunk['size'].tolist()[:500],
        'material': chunk['material'].tolist()[:500]
    }
    
    # Store embeddings in a dictionary
    embeddings = {}
    for field_name, texts in fields_to_embed.items():
        field_embeddings = llm_handler.generate_field_embeddings(texts, field_name)
        if field_embeddings is not None:
            embeddings[field_name] = field_embeddings
            print(f"Generated embeddings for {field_name}: shape {field_embeddings.shape}")
    
    return chunk, embeddings

def main():
    # Validate environment variables
    validate_config()
    
    # Process only one chunk for testing
    chunk_size = 10000
    print("Starting preprocessing...")
    
    # Initialize LLM handler with token from environment
    llm_handler = LLMHandler(HF_API_TOKEN)
    
    # Read only the first chunk
    chunk = next(pd.read_csv('products_train.csv', chunksize=chunk_size))
    processed_chunk, embeddings = preprocess_products(chunk, llm_handler)
    
    print("\nEmbedding results:")
    for field, emb in embeddings.items():
        print(f"{field}: {emb.shape if emb is not None else 'No embeddings generated'}")
    
    # Save the first chunk for verification
    processed_chunk.to_csv('preprocessed_sample.csv', index=False)
    print("\nPreprocessing completed! Sample saved to preprocessed_sample.csv")

if __name__ == "__main__":
    main() 