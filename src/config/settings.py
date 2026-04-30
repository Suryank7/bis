"""
Living Documentation Copilot - Configuration Settings

Centralized configuration management with environment variable support.
All settings are loaded once at startup and made available globally.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DocumentSettings:
    """Settings for document source and processing."""
    docs_path: Path = field(default_factory=lambda: Path(os.getenv("DOCS_PATH", "./data/docs")))
    supported_extensions: tuple = (".md", ".txt", ".pdf")
    

@dataclass
class ServerSettings:
    """Settings for the API server."""
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))


@dataclass
class LLMSettings:
    """Settings for the Language Model."""
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    
    # Ollama settings
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    
    # OpenAI settings (optional)
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


@dataclass
class EmbeddingSettings:
    """Settings for text embeddings."""
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    dimension: int = 384  # Dimension for all-MiniLM-L6-v2


@dataclass
class RAGSettings:
    """Settings for RAG pipeline."""
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "400")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    top_k_results: int = field(default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "5")))


@dataclass
class LoggingSettings:
    """Settings for logging."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


@dataclass
class Settings:
    """
    Main configuration container aggregating all settings.
    
    Usage:
        from src.config.settings import settings
        
        print(settings.documents.docs_path)
        print(settings.server.port)
        print(settings.llm.ollama_model)
    """
    documents: DocumentSettings = field(default_factory=DocumentSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    
    def __post_init__(self):
        """Ensure docs directory exists."""
        self.documents.docs_path.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        if self.llm.provider == "openai" and not self.llm.openai_api_key:
            errors.append("OpenAI provider selected but OPENAI_API_KEY not set")
        
        if self.rag.chunk_overlap >= self.rag.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        if errors:
            for error in errors:
                print(f"⚠️  Configuration Error: {error}")
            return False
        
        return True
    
    def print_config(self):
        """Print current configuration (hiding sensitive data)."""
        print("\n" + "=" * 60)
        print("🔧 LIVING DOCUMENTATION COPILOT - CONFIGURATION")
        print("=" * 60)
        print(f"📁 Documents Path:    {self.documents.docs_path.absolute()}")
        print(f"🌐 Server:            {self.server.host}:{self.server.port}")
        print(f"🤖 LLM Provider:      {self.llm.provider}")
        if self.llm.provider == "ollama":
            print(f"   └─ Model:          {self.llm.ollama_model}")
            print(f"   └─ Base URL:       {self.llm.ollama_base_url}")
        else:
            print(f"   └─ Model:          {self.llm.openai_model}")
            print(f"   └─ API Key:        {'****' + self.llm.openai_api_key[-4:] if self.llm.openai_api_key else 'NOT SET'}")
        print(f"📊 Embedding Model:   {self.embedding.model}")
        print(f"✂️  Chunk Size:        {self.rag.chunk_size} chars (overlap: {self.rag.chunk_overlap})")
        print(f"🔍 Top-K Results:     {self.rag.top_k_results}")
        print("=" * 60 + "\n")


# Global settings instance - import this everywhere
settings = Settings()
