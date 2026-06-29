"""Web search toolkit for agent."""

from typing import List, Dict, Any
from llama_cpp_agent.function_calling import LlamaCppFunctionTool


class WebSearchToolkit:
    """Pre-configured web search tools."""

    @staticmethod
    def search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results with title, url, snippet
        """
        try:
            from duckduckgo_search import DDGS

            ddgs = DDGS(timeout=10)
            results = ddgs.text(query, max_results=num_results)
            return results
        except ImportError:
            return [{"error": "duckduckgo-search not installed"}]

    @staticmethod
    def get_tools() -> List[LlamaCppFunctionTool]:
        """Return list of web search tools for agent registration."""
        return [
            LlamaCppFunctionTool(WebSearchToolkit.search),
        ]
