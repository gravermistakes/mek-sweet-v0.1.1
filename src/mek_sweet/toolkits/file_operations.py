"""File operations toolkit for agent."""

from typing import List, Dict, Any
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
import os


class FileOperationsToolkit:
    """Pre-configured file operations tools (sandboxed to working directory)."""

    @staticmethod
    def read_file(path: str) -> Dict[str, Any]:
        """Read file contents."""
        try:
            with open(path, "r") as f:
                content = f.read()
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def write_file(path: str, content: str) -> Dict[str, Any]:
        """Write content to file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return {"success": True, "message": f"Wrote {len(content)} bytes to {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def list_files(directory: str = ".") -> Dict[str, Any]:
        """List files in directory."""
        try:
            files = os.listdir(directory)
            return {"success": True, "files": files}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def get_tools() -> List[LlamaCppFunctionTool]:
        """Return list of file operation tools."""
        return [
            LlamaCppFunctionTool(FileOperationsToolkit.read_file),
            LlamaCppFunctionTool(FileOperationsToolkit.write_file),
            LlamaCppFunctionTool(FileOperationsToolkit.list_files),
        ]
