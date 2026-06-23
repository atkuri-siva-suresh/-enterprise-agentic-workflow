"""
Workflow state definition for the enterprise multi-agent system.

WorkflowState is the single source of truth passed between all nodes
in the LangGraph StateGraph. Each agent reads from and writes to this state.
"""
from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# All valid task types the supervisor can classify
TaskType = Literal["research", "document", "code", "report", "general"]

# All valid agent destinations
AgentName = Literal[
    "supervisor",
    "research_agent",
    "document_agent",
    "code_agent",
    "report_agent",
    "human_approval",
    "synthesizer",
    "__end__",
]


class TraceEntry(TypedDict):
    """Single entry in the agent execution trace — shown in the UI."""
    agent: str
    action: str
    input_summary: str
    output_summary: str
    timestamp: str
    duration_ms: int


class WorkflowState(TypedDict):
    """
    Shared state that flows through the entire LangGraph workflow.

    Design principles:
    - messages: append-only conversation history (LangGraph managed)
    - agent_outputs: keyed by agent name, stores each agent's last result
    - trace: append-only execution log for the UI dashboard
    - All routing decisions live in next_agent / task_type
    """

    # Full conversation history (append-only, managed by LangGraph)
    messages: Annotated[list[BaseMessage], add_messages]

    # Supervisor's classification of the incoming task
    task_type: Optional[TaskType]

    # Which agent should execute next (used for conditional routing)
    next_agent: Optional[AgentName]

    # Whether this task requires human approval before final output
    requires_approval: bool

    # Whether the human has approved (True/False/None = not yet requested)
    approved: Optional[bool]

    # Per-agent outputs — each agent writes its result here
    agent_outputs: dict

    # Execution trace for UI visualization
    trace: list

    # Final synthesized response
    final_response: Optional[str]

    # Cost tracking (approximate token counts)
    total_tokens_used: int

    # Number of agent hops taken (guards against infinite loops)
    iteration_count: int
