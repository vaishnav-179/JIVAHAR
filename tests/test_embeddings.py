import unittest
import numpy as np
from pathlib import Path
from ai_module.rag import DocumentChunk, DocumentProcessor, EmbeddingService

class TestEmbeddings(unittest.TestCase):
    """
    Unit test suite validating PDF document page-by-page chunking,
    sliding window overlap constraints, and local SentenceTransformers embedding generation.
    """
    
    def test_chunking_logic(self):
        # Use small sizes to easily test boundary overlaps
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        sample_text = "This is a simple text block to test sliding window overlaps."
        
        chunks = processor.split_text_into_chunks(sample_text, "test.pdf", page_num=2)
        
        self.assertGreater(len(chunks), 1, "Should split text into multiple chunks")
        for chunk in chunks:
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertEqual(chunk.source, "test.pdf")
            self.assertEqual(chunk.page, 2)
            self.assertTrue(len(chunk.text) <= 50, "Chunk text length should not exceed chunk_size")
            self.assertTrue(len(chunk.text) > 0, "Chunk text should not be empty")

    def test_process_knowledge_base_directory(self):
        kb_dir = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        chunks = processor.process_directory(kb_dir)
        
        self.assertGreater(len(chunks), 0, "Ingestion should yield at least one document chunk")
        
        # Verify that content from our manuals compiled in Phase 5 is present
        found_safety_terms = False
        for chunk in chunks:
            if "Danger Zone" in chunk.text or "Good Samaritan" in chunk.text or "NGO" in chunk.text:
                found_safety_terms = True
                break
        self.assertTrue(found_safety_terms, "Chunks should contain terms from safety/regulatory manuals")

    def test_embedding_generation(self):
        # Initialize the embedding service
        service = EmbeddingService()
        
        test_sentences = [
            "Surplus hot food should be Match-restricted to 5 km.",
            "Raw meat must be kept frozen at -18 degrees Celsius."
        ]
        
        # Generate batch embeddings
        embeddings = service.generate_embeddings(test_sentences)
        
        # Verify shapes and types
        self.assertIsInstance(embeddings, np.ndarray, "Batch embeddings must return a numpy array")
        self.assertEqual(embeddings.shape[0], 2, "Batch size mismatch")
        self.assertEqual(embeddings.shape[1], 384, "Embedding dimensions should be 384 (all-MiniLM-L6-v2 standard)")
        self.assertEqual(embeddings.dtype, np.float32, "FAISS database requires float32 datatype")

        # Test single query vector conversion
        query_vector = service.generate_query_embedding("Perishable food safety temperature")
        self.assertIsInstance(query_vector, np.ndarray)
        self.assertEqual(query_vector.shape, (384,), "Query vector must be 1D with 384 features")
        self.assertEqual(query_vector.dtype, np.float32)

if __name__ == "__main__":
    unittest.main()
