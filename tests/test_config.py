import unittest
from pathlib import Path
from config.settings import Settings

class TestConfig(unittest.TestCase):
    """
    Validates that the configuration management system operates correctly,
    properly loading environment variables and validating settings types.
    """
    
    def test_settings_loaded_from_env(self):
        try:
            # Instantiate settings (will parse the current .env)
            settings = Settings()
            
            # 1. Check API Key presence
            self.assertIsNotNone(settings.GEMINI_API_KEY, "API Key should load from .env")
            self.assertNotEqual(settings.GEMINI_API_KEY, "", "API Key should not be empty")
            
            # 2. Check defaults
            self.assertEqual(settings.GEMINI_MODEL, "gemini-2.5-flash")
            self.assertEqual(settings.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
            
            # 3. Check Path conversions and resolution
            self.assertIsInstance(settings.CNN_MODEL_PATH, Path)
            self.assertIsInstance(settings.FAISS_INDEX_PATH, Path)
            
            # Relative paths defined in .env should resolve to absolute paths on load
            self.assertTrue(settings.CNN_MODEL_PATH.is_absolute(), "Relative CNN_MODEL_PATH should be converted to absolute")
            self.assertTrue(settings.FAISS_INDEX_PATH.is_absolute(), "Relative FAISS_INDEX_PATH should be converted to absolute")
            
            # Log output to console during testing
            print("\n[CONFIG TEST] Configuration successfully loaded and validated:")
            print(f"  - GEMINI_MODEL: {settings.GEMINI_MODEL}")
            print(f"  - EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
            print(f"  - CNN_MODEL_PATH: {settings.CNN_MODEL_PATH}")
            print(f"  - FAISS_INDEX_PATH: {settings.FAISS_INDEX_PATH}")
            print(f"  - LOG_LEVEL: {settings.LOG_LEVEL}")
            
        except Exception as e:
            self.fail(f"Configuration class failed validation: {e}")

if __name__ == "__main__":
    unittest.main()
