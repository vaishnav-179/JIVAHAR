import os
import shutil
import unittest
from pathlib import Path
from PIL import Image
from unittest.mock import patch

from ai_module import IntegratedDonationPipeline
from ai_module.rag import RAGPipeline

class TestIntegratedDonationPipeline(unittest.TestCase):
    """
    End-to-end integration test validating the IntegratedDonationPipeline:
    instantiating models, loading CNN model checkpoint, running image preprocessing,
    vectorizing searches, formatting RAG contextual prompts, routing queries to mock Gemma,
    and returning a unified metadata payload.
    """
    
    def setUp(self):
        # Paths for temporary index storage
        self.test_index_dir = Path(__file__).resolve().parent / "temp_integrated_store"
        self.test_index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save a temporary dummy image to simulate photo uploads
        self.temp_image_path = Path(__file__).resolve().parent / "temp_pipeline_image.jpg"
        img = Image.new("RGB", (300, 300), color=(255, 0, 0))
        img.save(self.temp_image_path)

    def tearDown(self):
        # Remove temporary directories and test images
        if self.test_index_dir.exists():
            shutil.rmtree(self.test_index_dir)
        if self.temp_image_path.exists():
            os.remove(self.temp_image_path)

    @patch('ai_module.gemma.gemma_service.GemmaService.generate_response')
    def test_process_image_donation(self, mock_generate):
        # 1. Configure the GemmaService class method mock with conditional side-effects
        # Deliver distinct responses for summary generation vs safety assessments
        def generate_response_side_effect(prompt, system_instruction, temperature=0.1, max_output_tokens=1024):
            sys_inst_lower = system_instruction.lower()
            if "logistics" in sys_inst_lower:
                return (
                    "1. **Description**: Mocked description of food.\n"
                    "2. **Log Summary**: Food listed for donation.\n"
                    "3. **Logistics Recommendation**: Handle with care."
                )
            elif "safety" in sys_inst_lower:
                return (
                    "### Safety Assessment\nMocked safety assessment.\n\n"
                    "### Pickup Priority\n- Priority Score: HIGH\n- Justification: Mocked.\n\n"
                    "### Inspection Guidelines\n- Check smell."
                )
            return "Mocked response"
            
        mock_generate.side_effect = generate_response_side_effect

        # 2. Patch settings paths for clean database isolation
        with patch('ai_module.rag.vector_store.settings') as mock_settings:
            mock_settings.FAISS_INDEX_PATH = self.test_index_dir
            mock_settings.BASE_DIR = Path(__file__).resolve().parent.parent
            mock_settings.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock_settings.GEMINI_API_KEY = "dummy_key_for_pipeline"
            mock_settings.GEMINI_MODEL = "gemini-2.5-flash"
            mock_settings.CNN_MODEL_PATH = Path(__file__).resolve().parent.parent / "best_model.pth"
            
            # Ingest manual PDFs to build index files in test folder
            pipeline = RAGPipeline()
            pipeline.vector_store.index_file = self.test_index_dir / "index.faiss"
            pipeline.vector_store.meta_file = self.test_index_dir / "metadata.pkl"
            pipeline.ingest_documents()
            
            # 3. Initialize our integrated pipeline
            donation_pipeline = IntegratedDonationPipeline()
            
            # Override components index file paths to point to test folder
            donation_pipeline.safety_advisor.vector_store.index_file = self.test_index_dir / "index.faiss"
            donation_pipeline.safety_advisor.vector_store.meta_file = self.test_index_dir / "metadata.pkl"
            
            # 4. Execute integrated image donation processing
            payload = donation_pipeline.process_image_donation(
                image_path=str(self.temp_image_path),
                quantity="10 portions",
                prepared_time="1 hour ago",
                storage_condition="Refrigerated"
            )
            
            # 5. Assertions on payload keys and prediction outputs
            self.assertIn("food_name", payload)
            self.assertIn("cnn_confidence", payload)
            self.assertIn("summary", payload)
            self.assertIn("safety_advice", payload)
            self.assertIn("safety_sources", payload)
            
            # Verify CNN classification occurred (returns 'lassi' for solid colors)
            self.assertIsInstance(payload["food_name"], str)
            self.assertEqual(payload["food_name"], "lassi")
            self.assertGreater(payload["cnn_confidence"], 0.0)
            
            # Verify mock strings exist in summarization and safety attributes
            self.assertIn("Logistics Recommendation", payload["summary"])
            self.assertIn("Safety Assessment", payload["safety_advice"])
            
            # Verify RAG fetched matching citations
            self.assertGreater(len(payload["safety_sources"]), 0)
            self.assertTrue(any("food_safety_manual.pdf" == chunk.source for chunk in payload["safety_sources"]))

            # 6. Verify that Gemma generate_response was called exactly twice with appropriate parameters
            self.assertEqual(mock_generate.call_count, 2, "Gemma generate_response call count mismatch")
            
            # First call: DonationSummaryGenerator
            first_call_args, first_call_kwargs = mock_generate.call_args_list[0]
            self.assertIn("food redistribution logistics", first_call_kwargs['system_instruction'].lower())
            self.assertIn("lassi", first_call_kwargs['prompt'])
            self.assertIn("10 portions", first_call_kwargs['prompt'])
            
            # Second call: FoodSafetyAdvisor
            second_call_args, second_call_kwargs = mock_generate.call_args_list[1]
            self.assertIn("food safety advisor", second_call_kwargs['system_instruction'].lower())
            self.assertIn("lassi", second_call_kwargs['prompt'])
            self.assertIn("Refrigerated", second_call_kwargs['prompt'])

if __name__ == "__main__":
    unittest.main()
