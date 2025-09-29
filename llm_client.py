import os
import json
import google.generativeai as genai
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# An abstract interface for all LLM clients
class BaseLLMClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate_content(self, prompt: str) -> str:
        """
        The core method for generating content from a prompt.
        Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement this method")

# Concrete client for Gemini
class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        super().__init__(api_key, model)
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

    def generate_content(self, prompt: str) -> str:
        response = self.client.generate_content(prompt)
        return response.text.strip()

# Concrete client for OpenAI
class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key, model)
        self.client = OpenAI(api_key=self.api_key)

    def generate_content(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

# Concrete client for Groq
class GroqClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        super().__init__(api_key, model)
        self.client = Groq(api_key=self.api_key)

    def generate_content(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

# Factory function to create the right client
def create_llm_client(provider: str, api_key: str = None, model: str = None):
    """
    Factory function to create an LLM client for a specific provider.
    
    Args:
        provider: The name of the provider ('gemini', 'openai', 'groq').
        api_key: The API key for the provider. If None, it will be loaded from env.
        model: The specific model to use. If None, a default is used.

    Returns:
        An instance of an LLM client.
    
    Raises:
        ValueError: If the provider is not supported or API key is missing.
    """
    if provider.lower() == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found.")
        return GeminiClient(key, model or "gemini-2.5-flash")
    elif provider.lower() == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found.")
        return OpenAIClient(key, model or "gpt-4o")
    elif provider.lower() == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not found.")
        return GroqClient(key, model or "llama-3.1-8b-instant")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")