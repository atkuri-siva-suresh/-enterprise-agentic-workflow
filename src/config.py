"""
Configuration management for enterprise-agentic-workflow.
Loads from .env and provides typed settings.

Supported LLM providers (set LLM_PROVIDER in .env):
  - groq      (default, free tier at console.groq.com)
  - anthropic (requires paid API key)
  - ollama    (fully local, no API key needed)
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Provider → (main model, fast model)
PROVIDER_DEFAULTS = {
    "groq": (
        "llama-3.3-70b-versatile",   # Best quality on Groq free tier
        "llama-3.1-8b-instant",       # Fast model for routing/classification
    ),
    "anthropic": (
        "claude-opus-4-5",
        "claude-haiku-4-5-20251001",
    ),
    "ollama": (
        "llama3.2",                   # Run: ollama pull llama3.2
        "llama3.2",
    ),
}


class Config:
    # Provider selection
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()

    # API keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Model names (auto-set from provider defaults, overridable via .env)
    @classmethod
    def _model_defaults(cls):
        return PROVIDER_DEFAULTS.get(cls.LLM_PROVIDER, PROVIDER_DEFAULTS["groq"])

    @classmethod
    def get_model_name(cls) -> str:
        return os.getenv("MODEL_NAME") or cls._model_defaults()[0]

    @classmethod
    def get_fast_model_name(cls) -> str:
        return os.getenv("FAST_MODEL_NAME") or cls._model_defaults()[1]

    # LangSmith (optional observability)
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "enterprise-agentic-workflow")

    # App settings
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    APPROVAL_TIMEOUT_SECONDS: int = int(os.getenv("APPROVAL_TIMEOUT_SECONDS", "300"))

    # API server
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    @classmethod
    def validate(cls) -> None:
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is required for provider=groq.\n"
                "  1. Sign up free at https://console.groq.com\n"
                "  2. Create an API key (no credit card needed)\n"
                "  3. Add GROQ_API_KEY=gsk_... to your .env file"
            )
        if cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for provider=anthropic.\n"
                "  Get a key at https://console.anthropic.com"
            )

    @classmethod
    def is_langsmith_enabled(cls) -> bool:
        return cls.LANGCHAIN_TRACING_V2 and bool(cls.LANGCHAIN_API_KEY)


config = Config()
