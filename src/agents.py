"""
Agent node implementations for the enterprise multi-agent workflow.

Each agent is a pure function: (WorkflowState) -> dict of state updates.
Agents use Claude via LangChain's ChatAnthropic with bound tools.

Architecture:
    supervisor  →  [research | document | code | report]  →  synthesizer
                                     ↓
                            human_approval (if needed)
"""
import time
import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel

from .state import WorkflowState
from .tools import RESEARCH_TOOLS, DOCUMENT_TOOLS, CODE_TOOLS, REPORT_TOOLS
from .config import config


# ─── LLM Factory ──────────────────────────────────────────────────────────────

def _get_llm(fast: bool = False) -> BaseChatModel:
    """
    Provider-agnostic LLM factory.
    Reads LLM_PROVIDER from config and returns the appropriate chat model.
    Supports: groq (default), anthropic, ollama.
    """
    model = config.get_fast_model_name() if fast else config.get_model_name()
    provider = config.LLM_PROVIDER

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            api_key=config.GROQ_API_KEY,
            temperature=0,
            max_tokens=4096,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=config.ANTHROPIC_API_KEY,
            max_tokens=4096,
            temperature=0,
        )

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: '{provider}'. "
        "Choose from: groq, anthropic, ollama"
    )


# ─── Trace Helper ─────────────────────────────────────────────────────────────

def _make_trace_entry(agent: str, action: str, input_summary: str,
                      output_summary: str, duration_ms: int) -> dict:
    return {
        "agent": agent,
        "action": action,
        "input_summary": input_summary[:200],
        "output_summary": output_summary[:300],
        "timestamp": datetime.now().isoformat(),
        "duration_ms": duration_ms,
    }


# ─── Supervisor Node ──────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are the Supervisor Agent for an enterprise AI operations system.

Your job is to:
1. Analyze the user's request carefully
2. Classify the task type (one of: research, document, code, report, general)
3. Determine if human approval is required (for any action that modifies systems, sends communications, or makes financial decisions)
4. Route to the most appropriate specialist agent

Task Type Definitions:
- research: Finding information, market analysis, competitive intelligence, fact-finding
- document: Analyzing, summarizing, or extracting information from documents
- code: Code review, debugging, generation, architecture questions
- report: Creating structured business reports, executive summaries, status updates
- general: Conversational questions, clarifications, simple lookups

Approval Required When:
- Task involves sending any external communication
- Task involves modifying any system or database
- Task involves financial decisions above $1,000
- Task output will be used in a production system

Respond ONLY in this exact JSON format:
{
  "task_type": "<one of: research|document|code|report|general>",
  "next_agent": "<one of: research_agent|document_agent|code_agent|report_agent|synthesizer>",
  "requires_approval": <true|false>,
  "reasoning": "<one sentence explaining your routing decision>",
  "confidence": <0.0-1.0>
}"""


def supervisor_node(state: WorkflowState) -> dict:
    """
    Supervisor: classifies task and determines which agent to invoke next.
    Uses a fast model to minimize latency for this routing step.
    """
    start = time.time()
    llm = _get_llm(fast=True)

    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=f"User request: {last_human_msg}"),
    ]

    response = llm.invoke(messages)
    duration_ms = int((time.time() - start) * 1000)

    try:
        routing = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: extract JSON from response if wrapped in markdown
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        routing = json.loads(content)

    trace_entry = _make_trace_entry(
        agent="supervisor",
        action="classify_and_route",
        input_summary=last_human_msg,
        output_summary=f"→ {routing['next_agent']} | type: {routing['task_type']} | approval: {routing['requires_approval']}",
        duration_ms=duration_ms,
    )

    return {
        "task_type": routing["task_type"],
        "next_agent": routing["next_agent"],
        "requires_approval": routing["requires_approval"],
        "trace": state.get("trace", []) + [trace_entry],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


# ─── Agent Factory ────────────────────────────────────────────────────────────

def _run_agent_with_tools(
    agent_name: str,
    system_prompt: str,
    tools: list,
    state: WorkflowState,
) -> dict:
    """
    Generic agent runner that:
    1. Binds tools to the LLM
    2. Invokes the LLM with conversation history
    3. Executes any tool calls
    4. Returns the final response as state updates
    """
    start = time.time()
    llm = _get_llm(fast=False)
    llm_with_tools = llm.bind_tools(tools)

    tool_map = {t.name: t for t in tools}
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # Agentic loop: keep calling until no more tool calls
    max_loops = 5
    loop_count = 0
    new_messages = []

    while loop_count < max_loops:
        response = llm_with_tools.invoke(messages + new_messages)
        new_messages.append(response)

        if not response.tool_calls:
            break

        # Execute all tool calls
        for tool_call in response.tool_calls:
            tool_fn = tool_map.get(tool_call["name"])
            if tool_fn:
                try:
                    tool_result = tool_fn.invoke(tool_call["args"])
                except Exception as e:
                    tool_result = f"Tool error: {str(e)}"
            else:
                tool_result = f"Unknown tool: {tool_call['name']}"

            new_messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        loop_count += 1

    duration_ms = int((time.time() - start) * 1000)
    final_response = new_messages[-1].content if new_messages else "No response generated."

    trace_entry = _make_trace_entry(
        agent=agent_name,
        action="execute_with_tools",
        input_summary=state["messages"][-1].content if state["messages"] else "",
        output_summary=final_response,
        duration_ms=duration_ms,
    )

    # Determine if we need approval or can go straight to synthesizer
    next_agent = "human_approval" if state.get("requires_approval") else "synthesizer"

    return {
        "messages": new_messages,
        "agent_outputs": {**state.get("agent_outputs", {}), agent_name: final_response},
        "next_agent": next_agent,
        "trace": state.get("trace", []) + [trace_entry],
        "total_tokens_used": state.get("total_tokens_used", 0) + len(final_response.split()) * 2,
    }


# ─── Research Agent ───────────────────────────────────────────────────────────

RESEARCH_SYSTEM = """You are the Research Agent for an enterprise AI operations system.

Your specialty: finding, synthesizing, and presenting information with precision.

When given a research task:
1. Use the web_search tool to find relevant, current information
2. If specific documents are referenced, use fetch_document to retrieve them
3. Synthesize findings into a structured, executive-ready summary
4. Always cite your sources and indicate confidence levels
5. Flag any conflicting information or gaps in your research

Format your final response as:
## Research Summary: [Topic]

### Key Findings
[3-5 bullet points with the most important findings]

### Supporting Evidence
[Details and data points]

### Confidence Level
[High/Medium/Low] — [Brief explanation of why]

### Sources
[List of sources used]"""


def research_agent_node(state: WorkflowState) -> dict:
    """Research Agent: searches and synthesizes information."""
    return _run_agent_with_tools(
        agent_name="research_agent",
        system_prompt=RESEARCH_SYSTEM,
        tools=RESEARCH_TOOLS,
        state=state,
    )


# ─── Document Agent ───────────────────────────────────────────────────────────

DOCUMENT_SYSTEM = """You are the Document Analysis Agent for an enterprise AI operations system.

Your specialty: extracting insights, classifying intent, and summarizing complex documents.

When given a document analysis task:
1. Use classify_document_intent to understand what the document is about
2. Use extract_key_points to identify the most important information
3. If a URL is provided, use fetch_document first to retrieve the content
4. Provide a structured analysis with actionable insights

Format your final response as:
## Document Analysis: [Document Title/Type]

### Document Classification
[Type, intent, urgency]

### Key Insights
[Most important findings]

### Action Items
[Any required actions, owners, deadlines]

### Risk Flags
[Anything requiring immediate attention]"""


def document_agent_node(state: WorkflowState) -> dict:
    """Document Agent: analyzes and extracts insights from documents."""
    return _run_agent_with_tools(
        agent_name="document_agent",
        system_prompt=DOCUMENT_SYSTEM,
        tools=DOCUMENT_TOOLS,
        state=state,
    )


# ─── Code Agent ───────────────────────────────────────────────────────────────

CODE_SYSTEM = """You are the Code Review and Architecture Agent for an enterprise AI operations system.

Your specialty: reviewing code quality, identifying issues, and providing actionable improvement recommendations.

When given a code task:
1. Use analyze_code_quality to identify issues and metrics
2. Use search_code_patterns to find relevant best practices
3. Provide specific, actionable recommendations
4. Always consider security, performance, and maintainability

Format your final response as:
## Code Analysis: [File/Component Name]

### Quality Assessment
[Overall assessment with key metrics]

### Issues Found
[Critical / Warning / Info categories]

### Recommended Improvements
[Prioritized list with before/after examples where helpful]

### Security Considerations
[Any security-relevant findings]

### Next Steps
[Concrete action items]"""


def code_agent_node(state: WorkflowState) -> dict:
    """Code Agent: reviews code quality and provides recommendations."""
    return _run_agent_with_tools(
        agent_name="code_agent",
        system_prompt=CODE_SYSTEM,
        tools=CODE_TOOLS,
        state=state,
    )


# ─── Report Agent ─────────────────────────────────────────────────────────────

REPORT_SYSTEM = """You are the Report Generation Agent for an enterprise AI operations system.

Your specialty: creating polished, structured business reports ready for executive consumption.

When given a report task:
1. Use get_report_template to get the appropriate structure
2. Use format_data_as_table to format any tabular data cleanly
3. Fill the template with the provided information
4. Ensure the report is executive-ready: clear, concise, actionable

Report writing principles:
- Lead with the "so what" — why does this matter?
- Use data to support every major claim
- End with clear next steps and owners
- Use plain language — no jargon unless the audience is technical

Generate a complete, ready-to-use report in markdown format."""


def report_agent_node(state: WorkflowState) -> dict:
    """Report Agent: generates structured business reports."""
    return _run_agent_with_tools(
        agent_name="report_agent",
        system_prompt=REPORT_SYSTEM,
        tools=REPORT_TOOLS,
        state=state,
    )


# ─── Human Approval Node ──────────────────────────────────────────────────────

def human_approval_node(state: WorkflowState) -> dict:
    """
    Human-in-the-loop checkpoint.

    In API mode: pauses and waits for /approve or /reject endpoint call.
    In CLI mode: prompts the user directly.
    In UI mode: Streamlit shows an approval widget.

    This node does NOT call any LLM — it's a pure state checkpoint.
    """
    # Summarize what needs approval
    agent_outputs = state.get("agent_outputs", {})
    last_output = list(agent_outputs.values())[-1] if agent_outputs else "No output to review."

    trace_entry = _make_trace_entry(
        agent="human_approval",
        action="awaiting_human_review",
        input_summary=last_output[:200],
        output_summary="Pending human decision (approve/reject)",
        duration_ms=0,
    )

    return {
        "next_agent": "synthesizer",  # Will proceed to synthesizer after approval
        "trace": state.get("trace", []) + [trace_entry],
        # Note: 'approved' is set externally via API or UI
    }


# ─── Synthesizer Node ─────────────────────────────────────────────────────────

SYNTHESIZER_SYSTEM = """You are the Final Synthesizer for an enterprise AI operations system.

Your job: take all agent outputs and produce a single, polished, complete response.

Synthesis principles:
1. Combine all agent findings coherently — no repetition
2. Maintain executive-appropriate tone and structure
3. Add a brief "Confidence & Caveats" note if any information was uncertain
4. End with a clear "Next Steps" section

Do NOT say "According to the research agent..." or reveal internal agent names.
Present the output as a unified, authoritative response."""


def synthesizer_node(state: WorkflowState) -> dict:
    """
    Synthesizer: combines all agent outputs into the final response.
    This is always the last node before END.
    """
    start = time.time()
    llm = _get_llm(fast=False)

    agent_outputs = state.get("agent_outputs", {})

    if not agent_outputs:
        # No specialist agents ran — this was a general/simple query
        final_response = state["messages"][-1].content if state["messages"] else ""
        duration_ms = 0
    else:
        # Build synthesis prompt from all agent outputs
        outputs_text = "\n\n---\n\n".join([
            f"[{agent.upper()} OUTPUT]\n{output}"
            for agent, output in agent_outputs.items()
        ])

        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM),
            HumanMessage(content=f"Original request: {state['messages'][0].content if state['messages'] else 'Unknown'}"),
            HumanMessage(content=f"Agent outputs to synthesize:\n\n{outputs_text}"),
        ]

        response = llm.invoke(messages)
        final_response = response.content
        duration_ms = int((time.time() - start) * 1000)

    trace_entry = _make_trace_entry(
        agent="synthesizer",
        action="synthesize_final_response",
        input_summary=f"{len(agent_outputs)} agent outputs combined",
        output_summary=final_response[:200],
        duration_ms=duration_ms,
    )

    return {
        "final_response": final_response,
        "next_agent": "__end__",
        "trace": state.get("trace", []) + [trace_entry],
    }
