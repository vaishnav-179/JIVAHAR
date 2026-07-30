import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ai_module.rag import RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    """
    Integration test suite validating the unified RAGPipeline:
    ingesting PDF manuals, building local FAISS indexes, performing semantic search,
    formatting reference context blocks, and routing prompts to Gemma.
    """
    
    def setUp(self):
        # Create a temporary local path for pipeline database tests
        self.test_index_dir = Path(__file__).resolve().parent / "temp_rag_store"
        self.test_index_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up temporary test database folder
        if self.test_index_dir.exists():
            shutil.rmtree(self.test_index_dir)

    @patch('ai_module.rag.rag_pipeline.GemmaService')
    def test_rag_pipeline_flow(self, mock_gemma_class):
        # 1. Configure the GemmaService mock
        mock_gemma_instance = MagicMock()
        mock_gemma_class.return_value = mock_gemma_instance
        
        # Define mock response containing fake grounded citations
        mock_gemma_instance.generate_response.return_value = (
            "According to the Food Safety Manual (page 1), cooked food must not be left "
            "in the Temperature Danger Zone (4°C - 60°C) for more than 2 hours."
        )

        # 2. Patch configurations to direct index storage to our test folder
        with patch('ai_module.rag.vector_store.settings') as mock_settings:
            mock_settings.FAISS_INDEX_PATH = self.test_index_dir
            mock_settings.BASE_DIR = Path(__file__).resolve().parent.parent
            mock_settings.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock_settings.GEMINI_API_KEY = "dummy_key_for_pipeline_test"
            mock_settings.GEMINI_MODEL = "gemini-2.5-flash"
            
            # Initialize pipeline
            pipeline = RAGPipeline()
            
            # Point vector store files directly to our test folder
            pipeline.vector_store.index_file = self.test_index_dir / "index.faiss"
            pipeline.vector_store.meta_file = self.test_index_dir / "metadata.pkl"
            
            # 3. Test Ingestion
            # Runs DocumentProcessor on data/knowledge_base/ and compiles FAISS index in test folder
            pipeline.ingest_documents()
            
            self.assertTrue(pipeline.vector_store.index_file.exists(), "Index file was not saved")
            self.assertTrue(pipeline.vector_store.meta_file.exists(), "Metadata file was not saved")
            self.assertGreater(pipeline.vector_store.index.ntotal, 0, "No vectors were added to FAISS")
            
            # 4. Test Query with Context (Real search + Mocked LLM execution)
            query_str = "What is the temperature danger zone?"
            answer, source_chunks = pipeline.query_with_context(query_str, k=2)
            
            # Verify search hits and response
            self.assertIn("Temperature Danger Zone", answer)
            self.assertEqual(len(source_chunks), 2, "Retrieved context chunks size mismatch")
            
            # Verify retrieval pulled correct source manual
            found_safety_manual = any("food_safety_manual.pdf" == chunk.source for chunk in source_chunks)
            self.assertTrue(found_safety_manual, "Search failed to retrieve food_safety_manual.pdf details")

            # 5. Verify Prompt structure passed to GemmaService
            called_args, called_kwargs = mock_gemma_instance.generate_response.call_args
            
            # Check prompt payload
            user_prompt = called_kwargs['prompt']
            self.assertIn("--- Reference Document Context ---", user_prompt)
            self.assertIn("User Query: What is the temperature danger zone?", user_prompt)
            self.assertIn("food_safety_manual.pdf", user_prompt)
            
            # Check system instructions
            system_instruction = called_kwargs['system_instruction']
            self.assertIn("Jivahar RAG Advisor", system_instruction)
            self.assertIn("answer the user's query using ONLY the provided document context", system_instruction)

if __name__ == "__main__":
    unittest.main()
