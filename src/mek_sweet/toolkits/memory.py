"""Memory management toolkit for agent."""

from typing import List, Dict, Any
from llama_cpp_agent.function_calling import LlamaCppFunctionTool


class MemoryToolkit:
    """Pre-configured memory management tools."""

    def __init__(self):
        self.memories: Dict[str, str] = {}

    def store_memory(self, key: str, value: str) -> Dict[str, Any]:
        """Store a memory item."""
        self.memories[key] = value
        return {"success": True, "key": key, "stored": True}

    def recall_memory(self, key: str) -> Dict[str, Any]:
        """Recall a memory item."""
        if key in self.memories:
            return {"success": True, "key": key, "value": self.memories[key]}
        return {"success": False, "error": f"Memory key '{key}' not found"}

    def list_memories(self) -> Dict[str, Any]:
        """List all stored memories."""
        return {"success": True, "memories": list(self.memories.keys())}

    def clear_memories(self) -> Dict[str, Any]:
        """Clear all memories."""
        count = len(self.memories)
        self.memories.clear()
        return {"success": True, "cleared": count}

    def get_tools(self) -> List[LlamaCppFunctionTool]:
        """Return list of memory tools."""
        return [
            LlamaCppFunctionTool(self.store_memory),
            LlamaCppFunctionTool(self.recall_memory),
            LlamaCppFunctionTool(self.list_memories),
            LlamaCppFunctionTool(self.clear_memories),
        ]
