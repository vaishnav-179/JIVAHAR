import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ai_module.rag import FoodSafetyAdvisor, RAGPipeline

class TestFoodSafetyAdvisor(unittest.TestCase):
    """
    Integration test suite validating the FoodSafetyAdvisor component:
    ingesting document safety guidelines, retrieving context from FAISS,
    compiling the custom food safety prompt template, and executing Gemma inference.
    """
    
    def setUp(self):
        # Create a temporary local path for safety advisor index files
        self.test_index_dir = Path(__file__).resolve().parent / "temp_safety_store"
        self.test_index_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up temporary test database folder
        if self.test_index_dir.exists():
            shutil.rmtree(self.test_index_dir)

    @patch('ai_module.rag.safety_advisor.GemmaService')
    def test_get_safety_advice_flow(self, mock_gemma_class):
        # 1. Configure the GemmaService mock to return a structured safety advice block
        mock_gemma_instance = MagicMock()
        mock_gemma_class.return_value = mock_gemma_instance
        mock_gemma_instance.generate_response.return_value = (
            "### Safety Assessment\n"
            "The vegetable curry is safe to distribute, but has moderate risk because it was prepared 3 hours ago "
            "and stored at room temperature (exceeding the standard 2-hour room temperature danger zone limit).\n\n"
            "### Pickup Priority\n"
            "- Priority Score: HIGH\n"
            "- Justification: Food has been left in the temperature danger zone for over 2 hours.\n\n"
            "### Inspection Guidelines\n"
            "- Check for sour or acidic odor.\n"
            "- Look for visual discoloration or condensation inside container."
        )

        # 2. Patch configurations to direct index storage to our test folder
        with patch('ai_module.rag.vector_store.settings') as mock_settings:
            mock_settings.FAISS_INDEX_PATH = self.test_index_dir
            mock_settings.BASE_DIR = Path(__file__).resolve().parent.parent
            mock_settings.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock_settings.GEMINI_API_KEY = "dummy_key_for_safety_test"
            mock_settings.GEMINI_MODEL = "gemini-2.5-flash"
            
            # Build temp index from PDF manuals
            pipeline = RAGPipeline()
            pipeline.vector_store.index_file = self.test_index_dir / "index.faiss"
            pipeline.vector_store.meta_file = self.test_index_dir / "metadata.pkl"
            pipeline.ingest_documents()
            
            # Initialize FoodSafetyAdvisor
            advisor = FoodSafetyAdvisor()
            
            # Point advisor index file paths to our test folder
            advisor.vector_store.index_file = self.test_index_dir / "index.faiss"
            advisor.vector_store.meta_file = self.test_index_dir / "metadata.pkl"
            
            # 3. Query the safety advisor
            advice, source_chunks = advisor.get_safety_advice(
                food_name="Vegetable Curry",
                prepared_time="3 hours ago",
                storage_condition="Room Temperature"
            )
            
            # 4. Verify search hits and output
            self.assertIn("Safety Assessment", advice)
            self.assertIn("Priority Score: HIGH", advice)
            self.assertGreater(len(source_chunks), 0, "Safety advisor failed to retrieve context chunks")
            
            # Verify retrieval pulled food safety manuals
            found_safety_manual = any("food_safety_manual.pdf" == chunk.source for chunk in source_chunks)
            self.assertTrue(found_safety_manual, "Query search failed to retrieve food_safety_manual.pdf guidelines")

            # 5. Verify Prompt structure passed to GemmaService
            called_args, called_kwargs = mock_gemma_instance.generate_response.call_args
            
            # Check user prompt variables
            user_prompt = called_kwargs['prompt']
            self.assertIn("Food Item: Vegetable Curry", user_prompt)
            self.assertIn("Prepared Time: 3 hours ago", user_prompt)
            self.assertIn("Storage Condition: Room Temperature", user_prompt)
            self.assertIn("food_safety_manual.pdf", user_prompt)
            
            # Check system instructions
            system_instruction = called_kwargs['system_instruction']
            self.assertIn("expert ai food safety advisor", system_instruction.lower())
            self.assertIn("pickup priority", system_instruction.lower())

if __name__ == "__main__":
    unittest.main()
