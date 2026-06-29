"""
Mek-Sweet: Single-agent engagement harness with integrated toolkits and deterministic flows.

A high-level wrapper around llama-cpp-agent providing:
- Unified Agent interface (ECS-style coordination, non-agents)
- Pre-configured toolkits (web search, code execution, file ops, memory)
- Deterministic flow orchestration (AgentChain with error handling)
- Simplified provider setup and message formatting
"""

from llama_cpp_agent import (
    LlamaCppAgent,
    FunctionCallingAgent,
    StructuredOutputAgent,
    AgentChain,
    AgentChainElement,
    LlamaCppFunctionTool,
)

from .agent import MekSweetAgent, EngagementHarness
from .toolkits import (
    WebSearchToolkit,
    CodeInterpreterToolkit,
    FileOperationsToolkit,
    MemoryToolkit,
)
from .flows import DeterministicFlow, FlowTemplate, PresetFlows
from .providers import ProviderPreset, setup_provider

__version__ = "0.1.0"
__all__ = [
    "MekSweetAgent",
    "EngagementHarness",
    "WebSearchToolkit",
    "CodeInterpreterToolkit",
    "FileOperationsToolkit",
    "MemoryToolkit",
    "DeterministicFlow",
    "FlowTemplate",
    "PresetFlows",
    "ProviderPreset",
    "setup_provider",
    # Re-export core llama-cpp-agent classes
    "LlamaCppAgent",
    "FunctionCallingAgent",
    "StructuredOutputAgent",
    "AgentChain",
    "LlamaCppFunctionTool",
]
