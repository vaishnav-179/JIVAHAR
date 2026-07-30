import unittest
from unittest.mock import MagicMock, patch
from ai_module.gemma.gemma_service import GemmaService, GemmaConfigurationError, GemmaAPIError

class TestGemmaService(unittest.TestCase):
    """
    Unit tests for GemmaService. Validates mock-key protection,
    mocked API success pathways, and failure propagation wrapper logic.
    """
    
    @patch('ai_module.gemma.gemma_service.settings')
    def test_mock_key_fails_fast(self, mock_settings):
        # Ensure that using the placeholder mock key fails fast with a configuration error
        mock_settings.GEMINI_API_KEY = "mock_key_for_testing"
        mock_settings.GEMINI_MODEL = "gemini-2.5-flash"
        
        service = GemmaService()
        with self.assertRaises(GemmaConfigurationError) as context:
            service.generate_response("Hello Gemma")
            
        self.assertIn("configure a valid GEMINI_API_KEY", str(context.exception))

    @patch('ai_module.gemma.gemma_service.genai.Client')
    @patch('ai_module.gemma.gemma_service.settings')
    def test_successful_generation_mocked(self, mock_settings, mock_client_class):
        # Configure settings mock
        mock_settings.GEMINI_API_KEY = "real_looking_api_key_123"
        mock_settings.GEMINI_MODEL = "gemini-2.5-flash"
        
        # Configure client mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "This is a mocked response from the Gemma service."
        mock_client.models.generate_content.return_value = mock_response
        
        # Execute service
        service = GemmaService()
        result = service.generate_response(
            prompt="Hello, is this working?",
            system_instruction="You are a helpful assistant.",
            temperature=0.1
        )
        
        # Assertions
        self.assertEqual(result, "This is a mocked response from the Gemma service.")
        mock_client.models.generate_content.assert_called_once()
        
        # Verify the structure of the config argument passed to the client
        called_args, called_kwargs = mock_client.models.generate_content.call_args
        self.assertEqual(called_kwargs['model'], "gemini-2.5-flash")
        self.assertEqual(called_kwargs['contents'], "Hello, is this working?")
        
        config = called_kwargs['config']
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.system_instruction, "You are a helpful assistant.")

    @patch('ai_module.gemma.gemma_service.genai.Client')
    @patch('ai_module.gemma.gemma_service.settings')
    def test_api_failure_raises_custom_exception(self, mock_settings, mock_client_class):
        from google.genai.errors import APIError
        
        mock_settings.GEMINI_API_KEY = "real_looking_api_key_123"
        mock_settings.GEMINI_MODEL = "gemini-2.5-flash"
        
        # Configure client to raise APIError with the correct constructor signature
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # APIError constructor signature: APIError(code, response_json, response=None)
        mock_client.models.generate_content.side_effect = APIError(
            code=429,
            response_json={"error": {"message": "Resource has been exhausted (rate limit exceeded)"}}
        )
        
        service = GemmaService()
        with self.assertRaises(GemmaAPIError) as context:
            service.generate_response("Trigger an error")
            
        self.assertIn("API Error occurred", str(context.exception))
        self.assertIn("Resource has been exhausted", str(context.exception))

if __name__ == "__main__":
    unittest.main()
