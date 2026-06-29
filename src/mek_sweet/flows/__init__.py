"""
Deterministic flow orchestration for agent engagement.
"""

from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass, field
from llama_cpp_agent import AgentChain, AgentChainElement


@dataclass
class DeterministicFlow:
    """
    Orchestrates deterministic agent engagement flows.

    Wraps llama-cpp-agent's AgentChain with error handling and state management.
    """

    name: str
    description: str = ""
    steps: List[Callable] = field(default_factory=list)
    error_handling: str = "retry"  # or "skip", "stop"
    max_retries: int = 3

    def add_step(self, step: Callable) -> None:
        """Add a step to the flow."""
        self.steps.append(step)

    def execute(self, initial_input: Any, agent: Any) -> Dict[str, Any]:
        """Execute the flow with error handling."""
        results = []
        current = initial_input

        for i, step in enumerate(self.steps):
            try:
                result = step(current, agent)
                results.append(result)
                current = result
            except Exception as e:
                if self.error_handling == "stop":
                    return {
                        "success": False,
                        "step": i,
                        "error": str(e),
                        "results": results,
                    }
                elif self.error_handling == "skip":
                    continue
                else:  # retry
                    for _ in range(self.max_retries):
                        try:
                            result = step(current, agent)
                            results.append(result)
                            current = result
                            break
                        except Exception:
                            continue
                    else:
                        return {
                            "success": False,
                            "step": i,
                            "error": f"Failed after {self.max_retries} retries",
                            "results": results,
                        }

        return {
            "success": True,
            "steps_executed": len(results),
            "results": results,
            "final_output": current,
        }


@dataclass
class FlowTemplate:
    """Pre-configured flow templates."""

    @staticmethod
    def research_flow() -> DeterministicFlow:
        """Template: Search, analyze, summarize."""
        return DeterministicFlow(
            name="research",
            description="Search, analyze findings, generate summary",
            steps=[],
        )

    @staticmethod
    def coding_flow() -> DeterministicFlow:
        """Template: Analyze problem, write code, test."""
        return DeterministicFlow(
            name="coding",
            description="Understand problem, write solution, verify",
            steps=[],
        )

    @staticmethod
    def planning_flow() -> DeterministicFlow:
        """Template: Break down task, plan steps, estimate."""
        return DeterministicFlow(
            name="planning",
            description="Decompose goal, outline steps, estimate effort",
            steps=[],
        )


class PresetFlows:
    """Registry of pre-built flow templates."""

    RESEARCH = FlowTemplate.research_flow()
    CODING = FlowTemplate.coding_flow()
    PLANNING = FlowTemplate.planning_flow()
