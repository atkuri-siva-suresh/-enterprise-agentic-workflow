"""
Streamlit UI for the Enterprise Agentic Workflow.

Features:
- Task submission with real-time status indicators
- Agent trace visualization (which agents ran, how long, what they did)
- Human-in-the-loop approval widget
- Session history sidebar
- Token usage tracking
"""
import sys
import os
import time
import json
from datetime import datetime

import streamlit as st

# Add parent directory to path so src imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.workflow import run_workflow
from src.config import config

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise Agentic Workflow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .agent-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #7c3aed;
    }
    .agent-supervisor { border-left-color: #f59e0b; }
    .agent-research   { border-left-color: #3b82f6; }
    .agent-document   { border-left-color: #10b981; }
    .agent-code       { border-left-color: #ef4444; }
    .agent-report     { border-left-color: #8b5cf6; }
    .agent-approval   { border-left-color: #f97316; }
    .agent-synthesizer { border-left-color: #06b6d4; }

    .metric-badge {
        display: inline-block;
        background: #374151;
        color: #d1d5db;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        margin-right: 6px;
    }
    .status-running { color: #f59e0b; }
    .status-done { color: #10b981; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "task_input_value" not in st.session_state:
    st.session_state.task_input_value = ""


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Enterprise Agentic Workflow")
    st.caption(f"Session: `{st.session_state.thread_id}`")

    st.divider()

    # Agent Legend
    st.subheader("Agent Legend")
    agents_info = {
        "Supervisor": ("🎯", "#f59e0b", "Classifies & routes tasks"),
        "Research": ("🔍", "#3b82f6", "Web search & synthesis"),
        "Document": ("📄", "#10b981", "Document analysis"),
        "Code": ("💻", "#ef4444", "Code review & generation"),
        "Report": ("📊", "#8b5cf6", "Business report writing"),
        "Approval": ("✋", "#f97316", "Human checkpoint"),
        "Synthesizer": ("✨", "#06b6d4", "Final response builder"),
    }
    for name, (icon, color, desc) in agents_info.items():
        st.markdown(
            f'<span style="color:{color};">{icon} **{name}**</span> — {desc}',
            unsafe_allow_html=True
        )

    st.divider()

    # Session History
    st.subheader("Session History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            ts = item.get("timestamp", "")[:16]
            task_type = item.get("task_type", "?")
            msg_preview = item.get("message", "")[:40]
            st.caption(f"`{ts}` [{task_type}] {msg_preview}...")
    else:
        st.caption("No history yet. Run your first task!")

    st.divider()

    # Settings
    with st.expander("Settings"):
        new_session = st.button("New Session")
        if new_session:
            import uuid
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.session_state.history = []
            st.session_state.pending_approval = None
            st.rerun()

        st.caption(f"Model: `{config.get_model_name()}`")
        st.caption(f"LangSmith: {'Enabled' if config.is_langsmith_enabled() else 'Disabled'}")


# ─── Main Content ─────────────────────────────────────────────────────────────

st.title("Enterprise AI Operations Desk")
st.caption("Powered by LangGraph multi-agent orchestration and Claude")

# ── Example Tasks ─────────────────────────────────────────────────────────────
st.markdown("**Try an example task:**")

col1, col2, col3, col4 = st.columns(4)
example_tasks = {
    col1: ("Research", "Research the current state of LLMOps tooling and compare top 5 platforms"),
    col2: ("Document", "Analyze this contract: https://example.com/contract.pdf and flag risk clauses"),
    col3: ("Code", "Review this Python function for security issues:\n```python\ndef get_user(id):\n    return db.execute(f'SELECT * FROM users WHERE id={id}')\n```"),
    col4: ("Report", "Create an executive summary for Q4 AI initiative status: 3 projects on track, 1 delayed due to data quality issues, budget 85% utilized"),
}

for col, (label, task) in example_tasks.items():
    with col:
        if st.button(f"{label} Example", use_container_width=True):
            st.session_state["task_input"] = task
            st.rerun()

# ── Task Input ────────────────────────────────────────────────────────────────
st.divider()

user_input = st.text_area(
    "Enter your task or question",
    height=100,
    placeholder="Ask anything: research a topic, analyze a document, review code, generate a report...",
    key="task_input",
)

col_run, col_clear = st.columns([1, 4])
with col_run:
    run_button = st.button("Run Task", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=False):
        st.session_state["task_input"] = ""
        st.rerun()


# ─── Approval Widget ──────────────────────────────────────────────────────────

if st.session_state.pending_approval:
    pending = st.session_state.pending_approval
    st.warning("Human Approval Required")
    st.markdown(f"**Task:** {pending['message']}")

    with st.expander("Review Agent Output", expanded=True):
        st.markdown(pending.get("agent_output", "No output available."))

    col_approve, col_reject = st.columns(2)
    with col_approve:
        if st.button("Approve & Execute", type="primary", use_container_width=True):
            with st.spinner("Executing approved task..."):
                result = run_workflow(
                    user_message=pending["message"],
                    thread_id=st.session_state.thread_id,
                    approved=True,
                )
            st.session_state.pending_approval = None
            st.session_state.history.append({
                "message": pending["message"],
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "task_type": result.get("task_type"),
                "approved": True,
            })
            st.rerun()
    with col_reject:
        if st.button("Reject", use_container_width=True):
            st.session_state.pending_approval = None
            st.info("Task rejected. No action was taken.")
            st.rerun()


# ─── Run Workflow ─────────────────────────────────────────────────────────────

if run_button and user_input.strip():
    with st.spinner("Agents working..."):
        status_placeholder = st.empty()
        status_placeholder.info("Supervisor analyzing your task...")

        start_time = time.time()
        try:
            result = run_workflow(
                user_message=user_input.strip(),
                thread_id=st.session_state.thread_id,
            )
        except Exception as e:
            st.error(f"Workflow error: {str(e)}")
            st.stop()

        elapsed = time.time() - start_time
        status_placeholder.empty()

    # Save to history
    st.session_state.history.append({
        "message": user_input.strip(),
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "task_type": result.get("task_type"),
    })

    # Check for pending approval
    if result.get("requires_approval") and result.get("approved") is None:
        st.session_state.pending_approval = {
            "message": user_input.strip(),
            "agent_output": result.get("final_response", ""),
        }
        st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    st.success(f"Completed in {elapsed:.1f}s")

    tab_response, tab_trace, tab_debug = st.tabs(["Response", "Agent Trace", "Debug"])

    with tab_response:
        st.markdown("### Final Response")
        st.markdown(result.get("final_response", "No response generated."))

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Task Type", result.get("task_type", "—").title())
        with col_m2:
            st.metric("Agents Used", len(result.get("trace", [])))
        with col_m3:
            st.metric("~Tokens Used", f"{result.get('total_tokens_used', 0):,}")

    with tab_trace:
        st.markdown("### Agent Execution Trace")
        st.caption("Each card shows one agent step — in the order they executed.")

        trace = result.get("trace", [])
        if not trace:
            st.info("No trace data available.")
        else:
            agent_colors = {
                "supervisor": "#f59e0b",
                "research_agent": "#3b82f6",
                "document_agent": "#10b981",
                "code_agent": "#ef4444",
                "report_agent": "#8b5cf6",
                "human_approval": "#f97316",
                "synthesizer": "#06b6d4",
            }
            agent_icons = {
                "supervisor": "🎯",
                "research_agent": "🔍",
                "document_agent": "📄",
                "code_agent": "💻",
                "report_agent": "📊",
                "human_approval": "✋",
                "synthesizer": "✨",
            }

            for i, entry in enumerate(trace):
                agent = entry.get("agent", "unknown")
                color = agent_colors.get(agent, "#6b7280")
                icon = agent_icons.get(agent, "🤖")
                duration = entry.get("duration_ms", 0)

                with st.expander(
                    f"Step {i+1}: {icon} {agent.replace('_', ' ').title()} — {duration}ms",
                    expanded=i < 3
                ):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Action:** `{entry.get('action', '—')}`")
                        st.markdown(f"**Input:** {entry.get('input_summary', '—')}")
                    with col_b:
                        ts = entry.get("timestamp", "")[:19]
                        st.markdown(f"**Time:** `{ts}`")
                        st.markdown(f"**Duration:** `{duration}ms`")

                    st.markdown("**Output Preview:**")
                    st.text(entry.get("output_summary", "—")[:500])

    with tab_debug:
        st.markdown("### Raw State (for debugging)")
        st.json({
            "task_type": result.get("task_type"),
            "requires_approval": result.get("requires_approval"),
            "approved": result.get("approved"),
            "total_tokens_used": result.get("total_tokens_used"),
            "trace_count": len(result.get("trace", [])),
        })

elif run_button:
    st.warning("Please enter a task before running.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Enterprise Agentic Workflow • Built with LangGraph + Claude • "
    "github.com/atkuri-siva-suresh/enterprise-agentic-workflow"
)
