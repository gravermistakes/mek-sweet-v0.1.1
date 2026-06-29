"""
Unified agent abstraction for mek-sweet engagement harness.

Provides a single, opinionated Agent interface for single-agent engagement
with deterministic flows, integrated toolkits, and ECS-style coordination.
"""

from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass, field
from llama_cpp_agent import LlamaCppAgent, FunctionCallingAgent, AgentChain, AgentChainElement
from llama_cpp_agent.providers import LlmProvider, LlmSamplingSettings
from llama_cpp_agent.messages import ChatHistory, BasicChatHistory


@dataclass
class AgentConfig:
    """Configuration for a mek-sweet agent."""

    provider: LlmProvider
    system_prompt: str = "You are a helpful assistant."
    name: str = "Agent"
    max_retries: int = 3
    timeout_seconds: float = 30.0
    tools: List[Any] = field(default_factory=list)
    memory_enabled: bool = False
    sampling_settings: Optional[LlmSamplingSettings] = None


class MekSweetAgent:
    """
    Single unified agent with integrated toolkits and deterministic flows.

    Wraps llama-cpp-agent's FunctionCallingAgent with:
    - Retry logic and error handling
    - Tool management
    - Chat history
    - Optional memory
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = FunctionCallingAgent(
            model=config.provider,
            system_prompt=config.system_prompt,
            tools=config.tools or [],
            sampling_settings=config.sampling_settings,
        )
        self.chat_history: ChatHistory = BasicChatHistory()
        self.retry_count = 0

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a single task with the agent.

        Args:
            task: User input/task description
            context: Optional context dict (injected into system prompt)

        Returns:
            Dict with 'response', 'tools_used', 'success' keys
        """
        try:
            # Add context to prompt if provided
            system_prompt = self.config.system_prompt
            if context:
                system_prompt += f"\n\nContext: {context}"

            # Execute with retry logic
            for attempt in range(self.config.max_retries):
                try:
                    response = self.agent.get_chat_response(
                        messages=self.chat_history.messages,
                        user_message=task,
                        system_prompt=system_prompt,
                    )

                    self.chat_history.add_user_message(task)
                    self.chat_history.add_assistant_message(response)

                    return {
                        "response": response,
                        "tools_used": getattr(response, "tool_calls", []),
                        "success": True,
                        "attempts": attempt + 1,
                    }
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    self.retry_count += 1
                    continue

        except Exception as e:
            return {
                "response": None,
                "error": str(e),
                "success": False,
                "attempts": self.config.max_retries,
            }

    def add_tool(self, tool: Any) -> None:
        """Register a new tool with the agent."""
        if tool not in self.agent.tools:
            self.agent.tools.append(tool)

    def clear_history(self) -> None:
        """Clear chat history."""
        self.chat_history = BasicChatHistory()


class EngagementHarness:
    """
    ECS-style coordination for agent engagement flows.

    Manages task definitions, service registration, and deterministic workflows
    without spawning multiple agents.
    """

    def __init__(self, name: str):
        self.name = name
        self.agent: Optional[MekSweetAgent] = None
        self.flows: Dict[str, 'FlowDefinition'] = {}
        self.state: Dict[str, Any] = {}

    def register_agent(self, config: AgentConfig) -> None:
        """Register the primary agent."""
        self.agent = MekSweetAgent(config)

    def register_flow(self, flow_name: str, steps: List[Callable]) -> None:
        """Register a deterministic flow."""
        self.flows[flow_name] = FlowDefinition(flow_name, steps)

    def execute_flow(self, flow_name: str, initial_task: str) -> Dict[str, Any]:
        """Execute a registered flow."""
        if flow_name not in self.flows:
            raise ValueError(f"Flow '{flow_name}' not registered")

        if not self.agent:
            raise RuntimeError("No agent registered")

        flow = self.flows[flow_name]
        results = []
        current_input = initial_task

        for step in flow.steps:
            result = self.agent.run(current_input)
            results.append(result)

            if not result["success"]:
                return {
                    "flow": flow_name,
                    "success": False,
                    "completed_steps": len(results),
                    "error": result.get("error"),
                    "results": results,
                }

            # Use response as input to next step
            current_input = result["response"]

        return {
            "flow": flow_name,
            "success": True,
            "completed_steps": len(results),
            "results": results,
            "final_output": current_input,
        }


@dataclass
class FlowDefinition:
    """Definition of a deterministic engagement flow."""

    name: str
    steps: List[Callable] = field(default_factory=list)

    def add_step(self, step: Callable) -> None:
        """Add a step to the flow."""
        self.steps.append(step)
