"""Code interpreter toolkit for agent."""

from typing import List, Dict, Any
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
import subprocess


class CodeInterpreterToolkit:
    """Pre-configured code execution tools (Python only for safety)."""

    @staticmethod
    def execute_python(code: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute Python code safely with timeout.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Dict with 'output', 'error', 'returncode'
        """
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "output": result.stdout,
                "error": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Code execution timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_tools() -> List[LlamaCppFunctionTool]:
        """Return list of code execution tools."""
        return [
            LlamaCppFunctionTool(CodeInterpreterToolkit.execute_python),
        ]
