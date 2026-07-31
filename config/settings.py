import os
import logging
from pathlib import Path
from typing import ClassVar
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory of the Project (NEONEXUS)
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Validates and stores system-wide configuration settings loaded from environment variables and .env.
    Uses Pydantic V2 to enforce data types, defaults, and field validations.
    """
    
    # Path configuration
    BASE_DIR: ClassVar[Path] = BASE_DIR
    
    # Google AI Studio API Key (Obtain from: https://aistudio.google.com/)
    GEMINI_API_KEY: str = Field(..., description="Google AI Studio API Key")
    
    # Gemma / Gemini LLM settings
    GEMINI_MODEL: str = Field("gemini-flash-latest", description="Gemini/Gemma Model Name")
    
    # Embeddings model for RAG
    EMBEDDING_MODEL: str = Field("all-MiniLM-L6-v2", description="Local sentence-transformers embedding model name")
    
    # Storage paths
    FAISS_INDEX_PATH: Path = Field(default=Path("data/vector_store/faiss_index"), description="FAISS Vector Index Storage Directory")
    CNN_MODEL_PATH: Path = Field(default=Path("best_model.pth"), description="Path to PyTorch best_model.pth model file")
    
    # Logging configuration
    LOG_LEVEL: str = Field("INFO", description="Console Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # Unified Multi-backend LLM settings
    LLM_BACKEND: str = Field("gemini", description="LLM Backend: gemini, ollama, huggingface")
    OLLAMA_HOST: str = Field("http://localhost:11434", description="Ollama API server URL")
    HF_API_KEY: str = Field("", description="Hugging Face API key (optional)")

    # Load settings from .env file in the base directory
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("GEMINI_API_KEY")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped or v_stripped == "your_google_ai_studio_api_key_here" or v_stripped == "mock_key_for_testing":
            # Return placeholder so that local run modes (Ollama/HuggingFace) are not blocked from starting
            return "mock_key_for_testing"
        return v_stripped

    @field_validator("CNN_MODEL_PATH")
    @classmethod
    def validate_cnn_model_path(cls, v: Path) -> Path:
        # Resolve relative path against BASE_DIR
        if not v.is_absolute():
            v = BASE_DIR / v
        return v

    @field_validator("FAISS_INDEX_PATH")
    @classmethod
    def validate_faiss_path(cls, v: Path) -> Path:
        # Resolve relative path against BASE_DIR
        if not v.is_absolute():
            v = BASE_DIR / v
        return v

    def configure_logging(self) -> None:
        """
        Configures global logging configuration based on settings.
        """
        numeric_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

# Instantiate a single instance (Singleton pattern) to share configurations across files
try:
    settings = Settings()
    settings.configure_logging()
except Exception as e:
    # Print error since logger might not be fully configured yet
    print(f"Configuration Initialization Error: {e}")
    raise e
