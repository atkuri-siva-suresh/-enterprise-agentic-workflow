"""
LangGraph workflow definition for the enterprise multi-agent system.

Graph topology:
    START → supervisor → [research | document | code | report] → [human_approval?] → synthesizer → END

Routing logic lives in conditional edges via route_from_supervisor()
and route_after_specialist().
"""
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import WorkflowState
from .agents import (
    supervisor_node,
    research_agent_node,
    document_agent_node,
    code_agent_node,
    report_agent_node,
    human_approval_node,
    synthesizer_node,
)
from .config import config


# ─── Routing Functions ────────────────────────────────────────────────────────

def route_from_supervisor(state: WorkflowState) -> Literal[
    "research_agent", "document_agent", "code_agent", "report_agent", "synthesizer"
]:
    """
    Conditional edge: supervisor → specialist agent.
    Routes based on next_agent set by supervisor_node.
    """
    next_agent = state.get("next_agent", "synthesizer")

    # Guard against loop overflow
    if state.get("iteration_count", 0) >= config.MAX_ITERATIONS:
        return "synthesizer"

    valid_specialists = {
        "research_agent", "document_agent",
        "code_agent", "report_agent", "synthesizer"
    }

    return next_agent if next_agent in valid_specialists else "synthesizer"


def route_after_specialist(state: WorkflowState) -> Literal[
    "human_approval", "synthesizer"
]:
    """
    Conditional edge: specialist agent → [human_approval | synthesizer].
    Routes to human_approval if the supervisor flagged requires_approval=True.
    """
    if state.get("requires_approval") and not state.get("approved"):
        return "human_approval"
    return "synthesizer"


def route_after_approval(state: WorkflowState) -> Literal["synthesizer", END]:
    """
    Conditional edge: human_approval → [synthesizer | END].
    If the human rejected, we end the workflow without synthesizing.
    """
    if state.get("approved") is False:
        return END
    return "synthesizer"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_workflow(checkpointer=None) -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistent state
                      (e.g., MemorySaver for in-memory, SqliteSaver for disk)

    Returns:
        Compiled StateGraph ready for invocation
    """
    graph = StateGraph(WorkflowState)

    # ── Register Nodes ──────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("document_agent", document_agent_node)
    graph.add_node("code_agent", code_agent_node)
    graph.add_node("report_agent", report_agent_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("synthesizer", synthesizer_node)

    # ── Entry Point ──────────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Supervisor Routes to Specialist ─────────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research_agent": "research_agent",
            "document_agent": "document_agent",
            "code_agent": "code_agent",
            "report_agent": "report_agent",
            "synthesizer": "synthesizer",
        }
    )

    # ── Each Specialist Routes to Approval Check or Synthesizer ─────────────
    for specialist in ["research_agent", "document_agent", "code_agent", "report_agent"]:
        graph.add_conditional_edges(
            specialist,
            route_after_specialist,
            {
                "human_approval": "human_approval",
                "synthesizer": "synthesizer",
            }
        )

    # ── Approval Gate → Synthesizer or End ───────────────────────────────────
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "synthesizer": "synthesizer",
            END: END,
        }
    )

    # ── Synthesizer Always Ends ───────────────────────────────────────────────
    graph.add_edge("synthesizer", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    return graph.compile(checkpointer=checkpointer)


# ─── Default Workflow Instance ────────────────────────────────────────────────

# In-memory checkpointer for session persistence (survives multiple turns)
_memory = MemorySaver()

# Default compiled workflow — import this in API and UI
workflow = build_workflow(checkpointer=_memory)


# ─── Convenience Runner ───────────────────────────────────────────────────────

def run_workflow(
    user_message: str,
    thread_id: str = "default",
    approved: bool = None,
) -> dict:
    """
    High-level function to run the workflow with a user message.

    Args:
        user_message: The task or question from the user
        thread_id: Session identifier for state persistence
        approved: Pass True/False to set human approval status

    Returns:
        dict with final_response, trace, task_type, requires_approval
    """
    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "task_type": None,
        "next_agent": None,
        "requires_approval": False,
        "approved": approved,
        "agent_outputs": {},
        "trace": [],
        "final_response": None,
        "total_tokens_used": 0,
        "iteration_count": 0,
    }

    config_dict = {"configurable": {"thread_id": thread_id}}
    result = workflow.invoke(initial_state, config=config_dict)

    return {
        "final_response": result.get("final_response", ""),
        "trace": result.get("trace", []),
        "task_type": result.get("task_type"),
        "requires_approval": result.get("requires_approval", False),
        "approved": result.get("approved"),
        "total_tokens_used": result.get("total_tokens_used", 0),
    }
