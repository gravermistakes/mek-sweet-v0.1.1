# Mek-Sweet

Single-agent engagement harness with integrated toolkits and deterministic flows.

A high-level wrapper around [llama-cpp-agent](https://github.com/Maximilian-Winter/llama-cpp-agent) providing:

- **Unified Agent Interface** – ECS-style coordination for single agent engagement
- **Pre-configured Toolkits** – Web search, code execution, file operations, memory management
- **Deterministic Flows** – AgentChain orchestration with error handling and retry logic
- **Simplified Setup** – Provider presets and sensible defaults
- **Production-Ready** – Comprehensive examples and documentation

## Quick Start

### Installation

```bash
pip install mek-sweet

# With web search support
pip install mek-sweet[web-search]

# With all optional features
pip install mek-sweet[all]
```

### Basic Usage

```python
from mek_sweet import MekSweetAgent, AgentConfig, ProviderPreset, WebSearchToolkit

# Setup provider
provider = ProviderPreset.groq(api_key="your-groq-api-key")

# Create agent with tools
config = AgentConfig(
    provider=provider,
    system_prompt="You are a helpful research assistant",
    tools=WebSearchToolkit.get_tools(),
)

agent = MekSweetAgent(config)

# Run a task
result = agent.run("Find information about recent AI developments")
print(result["response"])
```

## Architecture

Mek-Sweet merges two components:

1. **llama-cpp-agent** (low-level framework)
   - Multi-agent types, provider abstractions, tool infrastructure
   - Vendored as `/src/llama_cpp_agent/`

2. **mek-sweet** (high-level harness)
   - Unified agent interface, pre-configured toolkits, deterministic flows
   - Located in `/src/mek_sweet/`

## Toolkits

- **WebSearchToolkit** – DuckDuckGo search integration
- **CodeInterpreterToolkit** – Safe Python execution
- **FileOperationsToolkit** – File I/O with sandboxing
- **MemoryToolkit** – Persistent memory across sessions

## Providers

Pre-configured LLM backends:

```python
from mek_sweet import setup_provider

# Local llama-cpp
provider = setup_provider("local", {"model_path": "./model.gguf"})

# Local server
provider = setup_provider("server", {"base_url": "http://localhost:8000"})

# Groq API
provider = setup_provider("groq", {"api_key": "your-key"})
```

## Advanced Usage

See [llama-cpp-agent docs](https://github.com/Maximilian-Winter/llama-cpp-agent) for:
- Multiple agent types
- Function calling and structured output
- RAG and memory systems
- 15+ message formatters
- Custom provider implementations

## License

MIT – See LICENSE file
