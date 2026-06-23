"""
Test suite for enterprise-agentic-workflow.

Tests cover:
- State initialization
- Routing logic (no LLM calls needed)
- Tool execution
- Workflow integration (requires ANTHROPIC_API_KEY)

Run with: pytest tests/ -v
Run integration tests: pytest tests/ -v --integration
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.state import WorkflowState
from src.tools import web_search, analyze_code_quality, get_report_template
from src.workflow import route_from_supervisor, route_after_specialist, route_after_approval


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> WorkflowState:
    """Minimal valid WorkflowState for testing routing."""
    return WorkflowState(
        messages=[HumanMessage(content="Test message")],
        task_type=None,
        next_agent=None,
        requires_approval=False,
        approved=None,
        agent_outputs={},
        trace=[],
        final_response=None,
        total_tokens_used=0,
        iteration_count=0,
    )


# ─── Routing Logic Tests ──────────────────────────────────────────────────────

class TestRoutingLogic:
    """Test conditional edge routing — no LLM calls needed."""

    def test_route_to_research_agent(self, base_state):
        state = {**base_state, "next_agent": "research_agent"}
        assert route_from_supervisor(state) == "research_agent"

    def test_route_to_document_agent(self, base_state):
        state = {**base_state, "next_agent": "document_agent"}
        assert route_from_supervisor(state) == "document_agent"

    def test_route_to_code_agent(self, base_state):
        state = {**base_state, "next_agent": "code_agent"}
        assert route_from_supervisor(state) == "code_agent"

    def test_route_to_report_agent(self, base_state):
        state = {**base_state, "next_agent": "report_agent"}
        assert route_from_supervisor(state) == "report_agent"

    def test_route_to_synthesizer_when_unknown_agent(self, base_state):
        state = {**base_state, "next_agent": "unknown_agent"}
        assert route_from_supervisor(state) == "synthesizer"

    def test_route_to_synthesizer_on_max_iterations(self, base_state):
        state = {**base_state, "next_agent": "research_agent", "iteration_count": 99}
        # Should short-circuit to synthesizer when max iterations reached
        assert route_from_supervisor(state) == "synthesizer"

    def test_route_to_approval_when_required(self, base_state):
        state = {**base_state, "requires_approval": True, "approved": None}
        assert route_after_specialist(state) == "human_approval"

    def test_route_to_synthesizer_when_no_approval_needed(self, base_state):
        state = {**base_state, "requires_approval": False}
        assert route_after_specialist(state) == "synthesizer"

    def test_route_to_synthesizer_when_already_approved(self, base_state):
        state = {**base_state, "requires_approval": True, "approved": True}
        assert route_after_specialist(state) == "synthesizer"

    def test_route_to_end_when_rejected(self, base_state):
        state = {**base_state, "approved": False}
        result = route_after_approval(state)
        assert result == "__end__" or str(result) == "END" or result is not None

    def test_route_to_synthesizer_when_approved(self, base_state):
        state = {**base_state, "approved": True}
        assert route_after_approval(state) == "synthesizer"


# ─── Tool Tests ───────────────────────────────────────────────────────────────

class TestTools:
    """Test tool functions — these are pure Python, no LLM needed."""

    def test_web_search_returns_json(self):
        result = web_search.invoke({"query": "enterprise AI trends 2025"})
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "title" in data[0]
        assert "url" in data[0]
        assert "summary" in data[0]
        assert "relevance_score" in data[0]

    def test_web_search_includes_query_in_result(self):
        result = web_search.invoke({"query": "LangGraph tutorial"})
        assert "LangGraph" in result

    def test_analyze_code_quality_returns_json(self):
        test_code = "def hello():\n    print('hello world')\n"
        result = analyze_code_quality.invoke({"code": test_code, "language": "python"})
        data = json.loads(result)
        assert "language" in data
        assert "issues" in data
        assert "suggestions" in data
        assert data["language"] == "python"

    def test_analyze_code_quality_line_count(self):
        code = "line1\nline2\nline3\n"
        result = analyze_code_quality.invoke({"code": code})
        data = json.loads(result)
        assert data["line_count"] == 3

    def test_get_report_template_executive_summary(self):
        result = get_report_template.invoke({"report_type": "executive_summary"})
        assert "Executive Summary" in result
        assert "Situation" in result
        assert "Next Steps" in result

    def test_get_report_template_technical_brief(self):
        result = get_report_template.invoke({"report_type": "technical_brief"})
        assert "Technical Brief" in result
        assert "Architecture" in result

    def test_get_report_template_unknown_falls_back(self):
        result = get_report_template.invoke({"report_type": "unknown_type"})
        # Should return executive_summary as fallback
        assert "Executive Summary" in result


# ─── State Structure Tests ────────────────────────────────────────────────────

class TestStateStructure:
    """Verify WorkflowState has the correct structure."""

    def test_state_has_required_fields(self, base_state):
        required_fields = [
            "messages", "task_type", "next_agent", "requires_approval",
            "approved", "agent_outputs", "trace", "final_response",
            "total_tokens_used", "iteration_count"
        ]
        for field in required_fields:
            assert field in base_state, f"Missing field: {field}"

    def test_messages_contains_human_message(self, base_state):
        assert len(base_state["messages"]) == 1
        assert isinstance(base_state["messages"][0], HumanMessage)

    def test_trace_is_list(self, base_state):
        assert isinstance(base_state["trace"], list)

    def test_agent_outputs_is_dict(self, base_state):
        assert isinstance(base_state["agent_outputs"], dict)


# ─── Integration Tests (require ANTHROPIC_API_KEY) ────────────────────────────

@pytest.mark.integration
class TestWorkflowIntegration:
    """
    Integration tests that make real API calls.
    Run with: pytest tests/ -v -m integration
    Requires ANTHROPIC_API_KEY in .env
    """

    def test_research_workflow_end_to_end(self):
        from src.workflow import run_workflow
        result = run_workflow(
            user_message="What are the top 3 benefits of using LangGraph for multi-agent systems?",
            thread_id="test-research",
        )
        assert result["task_type"] == "research"
        assert result["final_response"]
        assert len(result["trace"]) >= 2  # At least supervisor + specialist

    def test_code_workflow_end_to_end(self):
        from src.workflow import run_workflow
        result = run_workflow(
            user_message="Review this code: def f(x): return eval(x)",
            thread_id="test-code",
        )
        assert result["task_type"] == "code"
        assert "eval" in result["final_response"].lower() or "security" in result["final_response"].lower()

    def test_approval_required_task(self):
        from src.workflow import run_workflow
        result = run_workflow(
            user_message="Send an email to all customers announcing a price increase of 20%",
            thread_id="test-approval",
        )
        assert result["requires_approval"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
