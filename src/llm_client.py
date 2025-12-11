"""
Unified LLM client for multiple providers (OpenAI, Ollama).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Get project root and load env
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Supported models by provider
SUPPORTED_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "ollama": ["tinyllama", "gemma:2b", "gemma3:4b", "llama2", "mistral"]
}


class LLMClient:
    """Unified interface for multiple LLM providers."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider.lower()
        self.model = model
        self._validate()
        self._client = self._init_client()
    
    def _validate(self):
        """Validate provider and model combination."""
        if self.provider not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown provider: {self.provider}. Supported: {list(SUPPORTED_MODELS.keys())}")
    
    def _init_client(self) -> OpenAI:
        """Initialize the appropriate client based on provider."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found. Please set it in .env file.")
            return OpenAI(api_key=api_key)
        
        elif self.provider == "ollama":
            # Ollama uses OpenAI-compatible API at localhost:11434
            return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> str:
        """
        Generate a response using the configured model.
        
        Args:
            system_prompt: System/instruction prompt
            user_content: User message content
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def __repr__(self):
        return f"LLMClient(provider='{self.provider}', model='{self.model}')"


def list_available_models() -> dict:
    """Return dictionary of supported providers and their models."""
    return SUPPORTED_MODELS.copy()


if __name__ == "__main__":
    print("Testing LLM Client...\n")
    
    test_prompt = "You are a helpful assistant."
    test_question = "What is 2 + 2? Answer in one word."
    
    # Test OpenAI
    print("Testing OpenAI (gpt-4o-mini):")
    try:
        client = LLMClient("openai", "gpt-4o-mini")
        response = client.generate(test_prompt, test_question, temperature=0.3)
        print(f"  Response: {response}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
    
    # Test Ollama
    print("Testing Ollama (tinyllama):")
    try:
        client = LLMClient("ollama", "tinyllama")
        response = client.generate(test_prompt, test_question, temperature=0.3)
        print(f"  Response: {response}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
    
    print("Available models:", list_available_models())
