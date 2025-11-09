"""
Configuration settings for the Praxis AI backend.
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Praxis AI"
    VERSION: str = "1.0.0"
    
    # CORS Configuration
    CORS_ORIGINS: list[str] = ["http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:8080"]
    
    # LLM Configuration
    GROQ_API_KEY: Optional[str] = "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"  # Groq API key (fallback, prefer env variable GROQ_API_KEY)
    GEMINI_API_KEY: Optional[str] = None  # Will default to the provided key if not set in env
    HF_TOKEN: Optional[str] = "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"  # Hugging Face API token (fallback, prefer env variable HF_TOKEN)
    LLM_PROVIDER: str = "groq"  # "groq", "huggingface", "gemini", "openai", or "anthropic"
    LLM_MODEL: str = "llama-3.1-8b-instant"  # Groq model: "llama-3.1-8b-instant" (fast), "llama-3.1-70b-versatile" (if available), "mixtral-8x7b-32768", "gemma2-9b-it"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Using sentence-transformers for embeddings (free, local)
    
    # Vector Database Configuration
    VECTOR_DB_PROVIDER: str = "pinecone"  # "pinecone", "weaviate", "qdrant"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "praxis-ai-papers"
    
    # Graph Database Configuration
    GRAPH_DB_PROVIDER: str = "neo4j"  # "neo4j" or "arangodb"
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None
    
    # Research APIs
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None
    ARXIV_API_ENABLED: bool = True
    ENABLE_FULL_TEXT_SCRAPING: bool = True  # Enable full text scraping (slower but more comprehensive, needed for better KG links)
    ENABLE_CITATION_FOLLOWING: bool = True  # Enable citation following to find papers that cite the target paper
    
    # File Storage
    UPLOAD_DIR: str = "data/uploads"
    REPORTS_DIR: str = "data/reports"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Agent Configuration
    MAX_PAPERS_IN_CORPUS: int = 50
    MIN_PAPERS_IN_CORPUS: int = 20
    K_NEAREST_NEIGHBORS: int = 10
    NOVELTY_THRESHOLD_HIGH: float = 0.3
    NOVELTY_THRESHOLD_MEDIUM: float = 0.15
    
    # Augmentation Configuration
    ENABLE_AUGMENTATION: bool = True  # Enable augmentation process after initial analysis
    MAX_AUGMENTATION_PAPERS: int = 30  # Maximum additional papers to scrape during augmentation
    AUGMENTATION_PAPERS_PER_QUERY: int = 10  # Maximum papers per search query during augmentation
    
    # Cache Configuration
    REDIS_URL: Optional[str] = None
    ENABLE_CACHING: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REPORTS_DIR, exist_ok=True)
