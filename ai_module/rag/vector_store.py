import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss

from config.settings import settings
from ai_module.rag.document_processor import DocumentChunk
from ai_module.rag.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages FAISS flat L2 vector indexes and maps search hits back to 
    their original DocumentChunk structures using serialized metadata.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.index_dir: Path = settings.FAISS_INDEX_PATH
        self.index_file: Path = self.index_dir / "index.faiss"
        self.meta_file: Path = self.index_dir / "metadata.pkl"
        
        # In-memory caches
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[DocumentChunk] = []

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """
        Extracts texts from chunks, generates embeddings, builds a FAISS FlatL2 index,
        and saves both vector index and chunk metadata to disk.
        
        Args:
            chunks: List of DocumentChunk instances to index.
        """
        if not chunks:
            logger.warning("Empty chunk list provided. Skipping index build.")
            return

        logger.info(f"Building vector database index for {len(chunks)} chunks...")
        
        try:
            # 1. Generate text embeddings
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings(texts)
            
            # 2. Initialize FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            
            # 3. Add vectors to the index
            index.add(embeddings)
            
            # 4. Ensure storage directory exists
            self.index_dir.mkdir(parents=True, exist_ok=True)
            
            # 5. Persist index and metadata mapping to disk
            faiss.write_index(index, str(self.index_file))
            with open(self.meta_file, "wb") as f:
                pickle.dump(chunks, f)
                
            # Keep in-memory cache updated
            self.index = index
            self.chunks = chunks
            
            logger.info(f"Successfully built and persisted index with {index.ntotal} vectors to {self.index_file}")
            
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            raise e

    def load_index(self) -> bool:
        """
        Loads the FAISS index binary and metadata mapping pickle from disk.
        
        Returns:
            True if load succeeded, False if index files do not exist.
        """
        if not self.index_file.exists() or not self.meta_file.exists():
            logger.warning("FAISS index or metadata file not found on disk. Index needs to be built first.")
            return False
            
        try:
            logger.info(f"Loading FAISS index from {self.index_file}...")
            self.index = faiss.read_index(str(self.index_file))
            
            logger.info(f"Loading metadata mapping from {self.meta_file}...")
            with open(self.meta_file, "rb") as f:
                self.chunks = pickle.load(f)
                
            logger.info(f"Successfully loaded index containing {self.index.ntotal} vectors.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector index files from disk: {e}")
            raise e

    def search(self, query_text: str, k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Performs an L2 similarity search on the FAISS index to find the top K
        closest matching document chunks for a query string.
        
        Args:
            query_text: The search query string.
            k: Number of nearest neighbors to retrieve.
            
        Returns:
            A list of tuples: (DocumentChunk, L2_distance)
            L2 distance close to 0 indicates high similarity.
        """
        # Ensure database is loaded
        if self.index is None or not self.chunks:
            if not self.load_index():
                raise RuntimeError(
                    "Search failed: Vector index is not built and could not be loaded from disk. "
                    "Please run document ingestion to compile the database first."
                )

        try:
            # 1. Vectorize query text
            query_vector = self.embedding_service.generate_query_embedding(query_text)
            
            # 2. Reshape to 2D array (1, dimension) required by FAISS
            query_vector_2d = np.expand_dims(query_vector, axis=0)
            
            # 3. Perform similarity query
            # search returns: (distances_matrix, indices_matrix)
            distances, indices = self.index.search(query_vector_2d, k)
            
            # 4. Map index hits back to document chunks
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 if index has fewer elements than k
                    continue
                if idx < len(self.chunks):
                    results.append((self.chunks[idx], float(dist)))
                    
            logger.debug(f"Query '{query_text}' returned {len(results)} matches.")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise e
