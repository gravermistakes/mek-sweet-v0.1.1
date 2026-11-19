"""Behavioral regression tests for registered engagement flows.

Run from the repository root with:
    python -m unittest discover -s tests -v

The agent module is loaded with provider stubs so no model, credentials,
GPU, or network access are required to exercise orchestration.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock


def load_agent_module():
    dependency_names = [
        "llama_cpp_agent",
        "llama_cpp_agent.providers",
        "llama_cpp_agent.messages",
    ]
    saved = {name: sys.modules.get(name) for name in dependency_names}
    try:
        root = types.ModuleType("llama_cpp_agent")
        for name in ("LlamaCppAgent", "FunctionCallingAgent", "AgentChain", "AgentChainElement"):
            setattr(root, name, type(name, (), {}))
        providers = types.ModuleType("llama_cpp_agent.providers")
        for name in ("LlmProvider", "LlmSamplingSettings"):
            setattr(providers, name, type(name, (), {}))
        messages = types.ModuleType("llama_cpp_agent.messages")
        for name in ("ChatHistory", "BasicChatHistory"):
            setattr(messages, name, type(name, (), {}))
        sys.modules.update({
            "llama_cpp_agent": root,
            "llama_cpp_agent.providers": providers,
            "llama_cpp_agent.messages": messages,
        })
        path = Path(__file__).resolve().parents[1] / "src/mek_sweet/agent.py"
        spec = importlib.util.spec_from_file_location("_meksweet_agent_flow_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class EngagementFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_module = load_agent_module()

    def setUp(self):
        self.harness = self.agent_module.EngagementHarness("test")
        self.harness.agent = Mock()

    def test_steps_are_invoked_in_order_and_outputs_are_chained(self):
        calls = []

        def first(value, agent):
            self.assertIs(agent, self.harness.agent)
            calls.append("first")
            return value + "|one"

        def second(value, agent):
            calls.append("second")
            return value + "|two"

        self.harness.register_flow("sequence", [first, second])
        result = self.harness.execute_flow("sequence", "input")
        self.assertTrue(result["success"])
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(result["completed_steps"], 2)
        self.assertEqual(result["final_output"], "input|one|two")
        self.harness.agent.run.assert_not_called()

    def test_exception_stops_pipeline_without_running_following_steps(self):
        later = Mock()

        def fail(value, agent):
            raise ValueError("bad transformation")

        self.harness.register_flow("broken", [lambda value, agent: "ok", fail, later])
        result = self.harness.execute_flow("broken", "start")
        self.assertFalse(result["success"])
        self.assertEqual(result["completed_steps"], 1)
        self.assertEqual(result["results"], ["ok"])
        self.assertIn("bad transformation", result["error"])
        later.assert_not_called()

    def test_failure_result_stops_pipeline(self):
        later = Mock()
        self.harness.register_flow("failed", [
            lambda value, agent: {"success": False, "error": "step declined"},
            later,
        ])
        result = self.harness.execute_flow("failed", "start")
        self.assertFalse(result["success"])
        self.assertEqual(result["completed_steps"], 0)
        self.assertIn("step declined", result["error"])
        later.assert_not_called()


if __name__ == "__main__":
    unittest.main()
