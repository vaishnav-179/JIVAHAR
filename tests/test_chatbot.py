import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ai_module.rag import JivaharChatbot, RAGPipeline

class TestJivaharChatbot(unittest.TestCase):
    """
    Integration test suite validating the JivaharChatbot component:
    formatting chat history structures, executing semantic matches on FAISS FAQs,
    aggregating reference context, and compiling prompts for Gemma.
    """
    
    def setUp(self):
        # Create a temporary local path for chatbot index files
        self.test_index_dir = Path(__file__).resolve().parent / "temp_chat_store"
        self.test_index_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Clean up temporary test database folder
        if self.test_index_dir.exists():
            shutil.rmtree(self.test_index_dir)

    def test_history_formatting(self):
        # Verify formatting arrays are converted into standard dialogue scripts
        chatbot = JivaharChatbot()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! I am Jivahar Bot. How can I help you today?"},
            {"role": "user", "content": "How do I sign up?"}
        ]
        formatted = chatbot._format_history(history)
        expected = (
            "User: Hello\n"
            "Jivahar Bot: Hi! I am Jivahar Bot. How can I help you today?\n"
            "User: How do I sign up?"
        )
        self.assertEqual(formatted, expected, "Chronological chat history formatting mismatch")

    @patch('ai_module.rag.chatbot.GemmaService')
    def test_chatbot_flow(self, mock_gemma_class):
        # 1. Configure the GemmaService mock to return a grounded FAQ response
        mock_gemma_instance = MagicMock()
        mock_gemma_class.return_value = mock_gemma_instance
        mock_gemma_instance.generate_response.return_value = (
            "To list food, click 'Create Donation' in the donor dashboard, "
            "as described in Jivahar FAQs (Page 1)."
        )

        # 2. Patch configurations to direct index storage to our test folder
        with patch('ai_module.rag.vector_store.settings') as mock_settings:
            mock_settings.FAISS_INDEX_PATH = self.test_index_dir
            mock_settings.BASE_DIR = Path(__file__).resolve().parent.parent
            mock_settings.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock_settings.GEMINI_API_KEY = "dummy_key_for_chat_test"
            mock_settings.GEMINI_MODEL = "gemini-2.5-flash"
            
            # Build temp index from PDF manuals
            pipeline = RAGPipeline()
            pipeline.vector_store.index_file = self.test_index_dir / "index.faiss"
            pipeline.vector_store.meta_file = self.test_index_dir / "metadata.pkl"
            pipeline.ingest_documents()
            
            # Initialize JivaharChatbot
            chatbot = JivaharChatbot()
            
            # Point chatbot index file paths to our test folder
            chatbot.vector_store.index_file = self.test_index_dir / "index.faiss"
            chatbot.vector_store.meta_file = self.test_index_dir / "metadata.pkl"
            
            # Prior dialogue log
            history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi, how can I help you?"}
            ]
            
            # 3. Call Chat
            reply, source_chunks = chatbot.chat(
                user_message="How do I list food to donate?",
                chat_history=history
            )
            
            # 4. Verify RAG matches FAQ document
            self.assertIn("Jivahar FAQs", reply)
            self.assertGreater(len(source_chunks), 0, "Chatbot failed to retrieve context chunks")
            
            # Verify retrieval pulled platform FAQs
            found_faq_doc = any("platform_faqs.pdf" == chunk.source for chunk in source_chunks)
            self.assertTrue(found_faq_doc, "Chatbot query failed to retrieve platform_faqs.pdf details")

            # 5. Verify Prompt structure passed to GemmaService
            called_args, called_kwargs = mock_gemma_instance.generate_response.call_args
            
            # Check user prompt variables
            user_prompt = called_kwargs['prompt']
            self.assertIn("User: Hello\nJivahar Bot: Hi, how can I help you?", user_prompt)
            self.assertIn("User: How do I list food to donate?", user_prompt)
            self.assertIn("platform_faqs.pdf", user_prompt)
            
            # Check system instructions
            system_instruction = called_kwargs['system_instruction']
            self.assertIn("Jivahar Bot", system_instruction)
            self.assertIn("polite, helpful, and professional AI Chatbot", system_instruction)

if __name__ == "__main__":
    unittest.main()
