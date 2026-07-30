import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Handles local text vectorization using SentenceTransformers.
    Computes dense vector representations (embeddings) for document chunks 
    and search queries.
    """
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        try:
            logger.info(f"Loading local SentenceTransformer model: {self.model_name}...")
            # Automatically downloads (~90MB) on first run and caches locally
            self.model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise RuntimeError(f"Embedding initialization failed: {e}") from e

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates dense vector embeddings for a list of text strings.
        
        Args:
            texts: List of strings to encode.
            
        Returns:
            A numpy float32 ndarray of shape (num_texts, embedding_dimension).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
            
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            # Ensure float32 format required by FAISS
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise e

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generates a 1D vector embedding for a single query string.
        
        Args:
            query: The user search query.
            
        Returns:
            A 1D numpy float32 array of shape (embedding_dimension,).
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
            
        embeddings = self.generate_embeddings([query])
        return embeddings[0]
