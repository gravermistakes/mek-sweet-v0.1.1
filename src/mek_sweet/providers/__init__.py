"""
Simplified provider setup for mek-sweet.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from llama_cpp_agent.providers import (
    LlmProvider,
    LlmSamplingSettings,
    LlamaCppPython,
    LlamaCppServer,
    Groq,
)


@dataclass
class ProviderPreset:
    """Pre-configured provider presets."""

    @staticmethod
    def local_llama_cpp(
        model_path: str, n_gpu_layers: int = -1
    ) -> LlamaCppPython:
        """Configure local llama-cpp-python provider."""
        return LlamaCppPython(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
        )

    @staticmethod
    def llama_cpp_server(
        base_url: str = "http://localhost:8000",
    ) -> LlamaCppServer:
        """Configure llama.cpp server provider."""
        return LlamaCppServer(base_url=base_url)

    @staticmethod
    def groq(api_key: Optional[str] = None, model_name: str = "mixtral-8x7b-32768") -> Groq:
        """Configure Groq provider."""
        return Groq(api_key=api_key, model_name=model_name)


def setup_provider(
    provider_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> LlmProvider:
    """
    Setup an LLM provider with sensible defaults.

    Args:
        provider_type: 'local', 'server', 'groq'
        config: Provider-specific configuration

    Returns:
        Configured LlmProvider instance
    """
    config = config or {}

    if provider_type == "local":
        return ProviderPreset.local_llama_cpp(
            model_path=config.get("model_path", "./model.gguf"),
            n_gpu_layers=config.get("n_gpu_layers", -1),
        )
    elif provider_type == "server":
        return ProviderPreset.llama_cpp_server(
            base_url=config.get("base_url", "http://localhost:8000"),
        )
    elif provider_type == "groq":
        return ProviderPreset.groq(
            api_key=config.get("api_key"),
            model_name=config.get("model_name", "mixtral-8x7b-32768"),
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
