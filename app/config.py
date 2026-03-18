"""
Configuration module for the Endee AI Assistant backend.
Loads environment variables and provides typed settings.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Endee Vector Database
    ENDEE_URL: str = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
    ENDEE_AUTH_TOKEN: str = os.getenv("ENDEE_AUTH_TOKEN", "")

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # HuggingFace API Token (required for Inference API)
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Index Configuration
    INDEX_NAME: str = os.getenv("INDEX_NAME", "knowledge_base")
    INDEX_DIMENSION: int = int(os.getenv("INDEX_DIMENSION", "384"))

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Frontend URL (for CORS)
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")


settings = Settings()
