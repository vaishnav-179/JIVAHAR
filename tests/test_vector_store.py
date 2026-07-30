import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from ai_module.rag import DocumentChunk, VectorStore

class TestVectorStore(unittest.TestCase):
    """
    Unit test suite validating the complete FAISS VectorStore lifecycle:
    building database index, disk persistence, index reloading, 
    and semantic L2 nearest-neighbor retrieval.
    """
    
    def setUp(self):
        # Create a temporary local path for database tests
        self.test_index_dir = Path(__file__).resolve().parent / "temp_vector_store"
        self.test_index_dir.mkdir(parents=True, exist_ok=True)
        
        # Test document chunks representing distinct safety policies
        self.sample_chunks = [
            DocumentChunk(
                text="Surplus cooked meals must be matched to NGOs located within a 5 km radius of the donor.",
                source="ngo_policies.pdf",
                page=2
            ),
            DocumentChunk(
                text="Under the Good Samaritan Protection Act, donors are protected from civil liability for food donations.",
                source="government_regulations.pdf",
                page=1
            )
        ]

    def tearDown(self):
        # Clean up temporary test index directory
        if self.test_index_dir.exists():
            shutil.rmtree(self.test_index_dir)

    def test_vector_store_lifecycle(self):
        # Patch FAISS_INDEX_PATH to write to the temporary test folder
        with patch('ai_module.rag.vector_store.settings') as mock_settings:
            mock_settings.FAISS_INDEX_PATH = self.test_index_dir
            mock_settings.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            
            # Initialize VectorStore
            store = VectorStore()
            
            # Set target files paths on instance to match mocked directory
            store.index_file = self.test_index_dir / "index.faiss"
            store.meta_file = self.test_index_dir / "metadata.pkl"
            
            # 1. Build Index and Save to Disk
            store.build_index(self.sample_chunks)
            
            self.assertTrue(store.index_file.exists(), "FAISS index.faiss was not written to disk")
            self.assertTrue(store.meta_file.exists(), "Metadata metadata.pkl was not written to disk")
            
            # 2. Instantiate a fresh store and reload from disk
            fresh_store = VectorStore()
            fresh_store.index_file = self.test_index_dir / "index.faiss"
            fresh_store.meta_file = self.test_index_dir / "metadata.pkl"
            
            loaded = fresh_store.load_index()
            self.assertTrue(loaded, "Failed to load index files from disk")
            self.assertEqual(len(fresh_store.chunks), 2, "Loaded chunk metadata count mismatch")
            self.assertEqual(fresh_store.index.ntotal, 2, "Loaded vector index size mismatch")
            
            # 3. Test Query 1: Similarity Search for Good Samaritan rules
            results = fresh_store.search("Do donors face lawsuits or liability?", k=1)
            self.assertEqual(len(results), 1, "Should return exactly 1 nearest neighbor")
            
            matched_chunk, distance = results[0]
            self.assertEqual(matched_chunk.source, "government_regulations.pdf")
            self.assertEqual(matched_chunk.page, 1)
            self.assertIn("Good Samaritan Protection Act", matched_chunk.text)
            self.assertGreaterEqual(distance, 0.0, "L2 distance must be non-negative")
            
            # 4. Test Query 2: Proximity Match limits
            results2 = fresh_store.search("What is the maximum matching distance radius?", k=1)
            self.assertEqual(len(results2), 1)
            
            matched_chunk2, distance2 = results2[0]
            self.assertEqual(matched_chunk2.source, "ngo_policies.pdf")
            self.assertEqual(matched_chunk2.page, 2)
            self.assertIn("5 km radius", matched_chunk2.text)

if __name__ == "__main__":
    unittest.main()
