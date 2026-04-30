"""
Living Documentation Copilot - LLM Integration

Provides a unified interface to interact with LLMs using free options:
- Ollama (local, completely free)
- Fallback to mock responses for testing without LLM

No API keys required when using Ollama.
"""

import httpx
from typing import Optional, List, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    Ollama provides free, local LLM inference.
    Install from: https://ollama.ai
    Run: ollama pull llama3.2
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 60.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature: Creativity (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                return f"[LLM Error: {response.status_code}]"
                
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return "[LLM Error: Request timed out. Is Ollama running?]"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"[LLM Error: {str(e)}]"
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()


class MockLLM:
    """
    Mock LLM for testing without external dependencies.
    
    Returns simple pattern-matched responses based on context.
    Useful for development and testing.
    """
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Generate a mock response based on the context provided."""
        
        # Extract context from the prompt (look for common patterns)
        prompt_lower = prompt.lower()
        
        # Try to extract the actual question and context
        if "context:" in prompt_lower and "question:" in prompt_lower:
            # Split to find context and question
            parts = prompt.split("Question:", 1)
            if len(parts) > 1:
                context = parts[0].replace("Context:", "").strip()
                question = parts[1].strip()
                
                # Return a response that references the context
                return f"Based on the provided documentation: {context[:500]}..."
        
        return "I can answer questions based on the documentation. Please ensure documents are loaded."
    
    def is_available(self) -> bool:
        """Mock LLM is always available."""
        return True


class LLMClient:
    """
    Unified LLM client that automatically selects the best available option.
    
    Priority:
    1. Ollama (if running)
    2. Mock LLM (fallback for testing)
    
    Usage:
        llm = LLMClient.create()
        response = llm.generate("What is the vacation policy?")
    """
    
    def __init__(self, client: Any):
        self._client = client
        self._client_type = type(client).__name__
    
    @classmethod
    def create(
        cls,
        provider: str = "ollama",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2"
    ) -> "LLMClient":
        """
        Create an LLM client with the best available backend.
        
        Args:
            provider: Preferred provider ("ollama" or "mock")
            ollama_base_url: Ollama API URL
            ollama_model: Model to use with Ollama
            
        Returns:
            Configured LLMClient instance
        """
        if provider == "ollama":
            client = OllamaClient(ollama_base_url, ollama_model)
            if client.is_available():
                logger.info(f"✅ Using Ollama LLM: {ollama_model}")
                return cls(client)
            else:
                logger.warning("⚠️ Ollama not available, falling back to mock LLM")
                logger.warning("   Install Ollama from https://ollama.ai and run: ollama pull llama3.2")
        
        logger.info("📝 Using Mock LLM (for testing)")
        return cls(MockLLM())
    
    @property
    def client_type(self) -> str:
        """Get the type of client being used."""
        return self._client_type
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Generate a response from the LLM."""
        return self._client.generate(prompt, system_prompt, temperature, max_tokens)
    
    def is_available(self) -> bool:
        """Check if the LLM backend is available."""
        return self._client.is_available()


# Export public API
__all__ = [
    "OllamaClient",
    "MockLLM",
    "LLMClient",
]
