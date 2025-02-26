import numpy as np
import requests
import time
from typing import List, Optional, Dict, Union
import os
from pathlib import Path
import pandas as pd
from config import HF_MODEL_NAME

class LLMHandler:
    def __init__(self, api_token: str, model_name: str = HF_MODEL_NAME):
        """
        Initialize the LLM handler
        
        Args:
            api_token: Hugging Face API token
            model_name: Name of the model to use
        """
        self.api_token = api_token
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        
        # Create embeddings directory if it doesn't exist
        self.embeddings_dir = Path("embeddings")
        self.embeddings_dir.mkdir(exist_ok=True)
    
    def get_embeddings(
        self,
        texts: List[str],
        max_retries: int = 3,
        batch_size: int = 100,
        prefix: str = "query"
    ) -> Optional[np.ndarray]:
        """
        Get embeddings for a list of texts using batching and retries
        
        Args:
            texts: List of texts to get embeddings for
            max_retries: Maximum number of retries for failed requests
            batch_size: Number of texts to process in each batch
            prefix: Prefix to add to each text ('query' or 'passage')
        
        Returns:
            Numpy array of embeddings if successful, None otherwise
        """
        # Add prefix to each text
        texts = [f"{prefix}: {text}" if text else "" for text in texts]
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return None
        
        all_embeddings = []
        
        # Process in smaller batches
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json={"inputs": batch_texts}
                    )
                    
                    if response.status_code == 200:
                        batch_embeddings = np.array(response.json())
                        all_embeddings.append(batch_embeddings)
                        print(f"Processed batch {i//batch_size + 1}/{len(valid_texts)//batch_size + 1}")
                        break
                    elif response.status_code == 503:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5  # Exponential backoff
                            print(f"Service unavailable, waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(f"Failed to process batch after {max_retries} attempts")
                            return None
                    else:
                        print(f"Error: {response.status_code} - {response.text}")
                        return None
                        
                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(5)
            
            # Small delay between batches to avoid rate limits
            time.sleep(1)
        
        if not all_embeddings:
            return None
            
        # Combine all batch embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
        normalized_embeddings = combined_embeddings / norms
        
        return normalized_embeddings
    
    def generate_field_embeddings(
        self,
        texts: List[str],
        field_name: str,
        save: bool = True
    ) -> Optional[np.ndarray]:
        """
        Generate embeddings for a specific field with error handling
        
        Args:
            texts: List of texts to generate embeddings for
            field_name: Name of the field (used for saving)
            save: Whether to save the embeddings to a file
        
        Returns:
            Numpy array of embeddings if successful, None otherwise
        """
        try:
            # Remove None and empty strings
            valid_texts = [str(t) for t in texts if pd.notna(t) and str(t).strip()]
            if not valid_texts:
                print(f"No valid texts found for field: {field_name}")
                return None
                
            print(f"Generating embeddings for {field_name} ({len(valid_texts)} items)")
            embeddings = self.get_embeddings(valid_texts)
            
            if embeddings is not None and save:
                # Save embeddings to file
                save_path = self.embeddings_dir / f"embeddings_{field_name}.npy"
                np.save(save_path, embeddings)
                print(f"Saved embeddings to {save_path}")
                
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings for {field_name}: {str(e)}")
            return None
    
    def compute_similarity(
        self,
        query_embeddings: np.ndarray,
        passage_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and passage embeddings
        
        Args:
            query_embeddings: Query embeddings matrix
            passage_embeddings: Passage embeddings matrix
        
        Returns:
            Similarity scores matrix
        """
        return np.dot(query_embeddings, passage_embeddings.T) * 100 