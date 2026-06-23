# Enterprise Agentic Workflow

> **Production-grade multi-agent AI orchestration for enterprise operations.**  
> Built with LangGraph + Anthropic Claude. Deployable in under 10 minutes.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Claude](https://img.shields.io/badge/Claude-claude--opus--4-orange.svg)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem This Solves

Enterprise clients don't just want AI to *answer questions* — they want it to *act*.  
But "autonomous AI that acts" needs three things most demos skip:

1. **Intelligent routing** — different tasks need different specialist agents  
2. **Human-in-the-loop gates** — before any action crosses a threshold, a human approves  
3. **Full observability** — every agent hop is traced, timed, and auditable

This system packages those three patterns into a production-ready reference implementation.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │         Enterprise Task Request          │
                        └──────────────────┬──────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │            SUPERVISOR AGENT              │
                        │   • Classifies task type                 │
                        │   • Detects approval requirements        │
                        │   • Routes to specialist agent           │
                        └────┬──────┬────────┬────────┬───────────┘
                             │      │        │        │
               ┌─────────────┘      │        │        └──────────────────┐
               │                    │        │                           │
               ▼                    ▼        ▼                           ▼
    ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
    │  RESEARCH AGENT  │  │ DOCUMENT     │  │  CODE AGENT  │  │  REPORT AGENT    │
    │                  │  │    AGENT     │  │              │  │                  │
    │ • Web search     │  │ • Key points │  │ • Quality    │  │ • Templates      │
    │ • Synthesis      │  │ • Intent     │  │   analysis   │  │ • Data tables    │
    │ • Source citing  │  │ • Risk flags │  │ • Security   │  │ • Exec summaries │
    └────────┬─────────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
             │                   │                  │                   │
             └───────────────────┴──────────────────┴───────────────────┘
                                           │
                          ┌────────────────▼─────────────────┐
                          │       HUMAN APPROVAL GATE         │
                          │   (only for high-stakes actions)  │
                          │   Approve via UI / API / CLI      │
                          └────────────────┬─────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │            SYNTHESIZER AGENT             │
                        │   • Combines all agent outputs           │
                        │   • Produces executive-ready response    │
                        │   • Adds confidence & next steps         │
                        └─────────────────────────────────────────┘
```

**State flows through the entire graph** — every agent reads from and writes to a shared `WorkflowState`. The LangGraph `MemorySaver` checkpointer enables multi-turn conversations across the same thread.

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/atkuri-siva-suresh/enterprise-agentic-workflow.git
cd enterprise-agentic-workflow

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run (choose your interface)

**Option A — Streamlit UI (recommended for first run)**
```bash
streamlit run ui/app.py
# Opens at http://localhost:8501
```

**Option B — FastAPI**
```bash
uvicorn api.main:app --reload
# Opens at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

**Option C — Demo script (CLI)**
```bash
python examples/demo.py --task research
python examples/demo.py --task code
python examples/demo.py --task all
```

**Option D — Docker Compose**
```bash
docker-compose up
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## Usage Examples

### Research Task
```python
from src.workflow import run_workflow

result = run_workflow(
    user_message="Compare LangGraph vs AutoGen for enterprise multi-agent deployments. Include production considerations.",
    thread_id="session-001",
)

print(result["final_response"])
print(f"Agents used: {len(result['trace'])}")
print(f"Task type: {result['task_type']}")
```

### Code Review
```python
result = run_workflow(
    user_message="""
    Review this function for security issues:
    def get_user(id):
        return db.execute(f'SELECT * FROM users WHERE id={id}')
    """,
    thread_id="session-002",
)
```

### Report Generation with Approval Gate
```python
# This task will trigger requires_approval=True
result = run_workflow(
    user_message="Send Q4 performance report to all client stakeholders",
    thread_id="session-003",
)

if result["requires_approval"]:
    # Review agent output first
    print(result["final_response"])

    # Approve via API:
    # POST /approve  {"run_id": "...", "approved": true}

    # Or re-run with approval:
    final = run_workflow(
        user_message="Send Q4 performance report to all client stakeholders",
        thread_id="session-003",
        approved=True,
    )
```

### API Usage
```bash
# Submit a task
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"message": "Research the top 5 LLMOps platforms for 2025"}'

# Stream agent trace in real-time
curl "http://localhost:8000/run/stream?message=Analyze+this+code..."

# Check pending approvals
curl http://localhost:8000/pending-approvals

# Approve a task
curl -X POST http://localhost:8000/approve \
  -H "Content-Type: application/json" \
  -d '{"run_id": "abc12345", "approved": true}'
```

---

## Agent Capabilities

| Agent | Specialty | Tools Used | When Triggered |
|-------|-----------|-----------|----------------|
| **Supervisor** | Task classification & routing | — | Always first |
| **Research Agent** | Web search, fact synthesis | `web_search`, `fetch_document` | Information gathering tasks |
| **Document Agent** | Document analysis, extraction | `extract_key_points`, `classify_document_intent` | Document analysis tasks |
| **Code Agent** | Code review, security analysis | `analyze_code_quality`, `search_code_patterns` | Code-related tasks |
| **Report Agent** | Business report generation | `get_report_template`, `format_data_as_table` | Report/summary tasks |
| **Human Approval** | Approval checkpoint | — | High-stakes actions |
| **Synthesizer** | Final response composition | — | Always last |

---

## Human-in-the-Loop Design

Approval is triggered automatically when the Supervisor detects:
- Tasks that send external communications
- Tasks that modify systems or databases
- Tasks with financial decisions above threshold
- Tasks explicitly flagged as production-destined

```
┌─────────────────────────────────────────────────────────────┐
│                  APPROVAL FLOW                               │
│                                                             │
│  Agent output ready → Supervisor flags requires_approval    │
│         │                                                   │
│         ▼                                                   │
│  Human reviews output in Streamlit UI / API response        │
│         │                                                   │
│         ├── Approve ──→ Synthesizer runs, task completes    │
│         └── Reject  ──→ Workflow ends, no action taken      │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
enterprise-agentic-workflow/
├── src/
│   ├── config.py      # Configuration management (API keys, model settings)
│   ├── state.py       # WorkflowState TypedDict (shared state schema)
│   ├── tools.py       # Tool definitions (web_search, analyze_code_quality, etc.)
│   ├── agents.py      # All agent node implementations
│   └── workflow.py    # LangGraph StateGraph definition + routing logic
├── api/
│   └── main.py        # FastAPI app (REST + SSE streaming endpoints)
├── ui/
│   └── app.py         # Streamlit dashboard with agent trace visualization
├── examples/
│   └── demo.py        # CLI demo script (Rich terminal output)
├── tests/
│   └── test_workflow.py  # Pytest unit + integration tests
├── .env.example       # Environment variable template
├── docker-compose.yml # Docker deployment
└── requirements.txt
```

---

## Extending the System

### Adding a New Specialist Agent

```python
# 1. Define tools in src/tools.py
@tool
def your_new_tool(param: str) -> str:
    """Tool description."""
    return "tool result"

YOUR_TOOLS = [your_new_tool]

# 2. Add agent node in src/agents.py
def your_agent_node(state: WorkflowState) -> dict:
    return _run_agent_with_tools(
        agent_name="your_agent",
        system_prompt="You are a specialist in X...",
        tools=YOUR_TOOLS,
        state=state,
    )

# 3. Register in src/workflow.py
graph.add_node("your_agent", your_agent_node)
graph.add_conditional_edges("supervisor", route_from_supervisor, {
    "your_agent": "your_agent",
    # ... existing routes
})
```

### Connecting to Real Systems

The tool stubs in `src/tools.py` are designed to be swapped for real integrations:

```python
# Replace the mock body in web_search with Tavily:
from tavily import TavilyClient
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
response = client.search(query=query)

# Replace fetch_document with a real document parser:
import httpx
from bs4 import BeautifulSoup
r = httpx.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
return soup.get_text()
```

### Enabling LangSmith Observability

```env
# In .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-key
LANGCHAIN_PROJECT=enterprise-agentic-workflow
```

Every agent invocation will appear in your LangSmith dashboard with full token counts, latency, and input/output traces.

---

## Tests

```bash
# Unit tests (no API key needed)
pytest tests/ -v

# Integration tests (requires ANTHROPIC_API_KEY)
pytest tests/ -v -m integration
```

---

## Production Lessons

This system was designed from real patterns learned deploying enterprise AI at scale:

1. **Approval gates prevent 80% of "AI did something unexpected" complaints** — Human checkpoints are not UX friction; they're trust builders. Enterprise clients adopt AI faster when they can see and approve before anything irreversible happens.

2. **Structured state over free-form messages** — Using TypedDict state instead of passing raw message lists makes debugging 10x easier. You can print the state at any node and know exactly what happened.

3. **Fast model for routing, full model for generation** — The supervisor uses a fast model (Haiku) to classify and route. Only specialist agents use the full model. This cuts cost and latency without sacrificing quality.

4. **Tool call loops need guards** — Every agent loop has a `max_loops=5` guard. In production, LLM tool use can spiral. Always cap iterations.

5. **Trace everything from day one** — The trace list in WorkflowState was added after the first production deployment. Debugging a 5-hop agent chain without traces is painful. Every agent writes a trace entry — no exceptions.

---

## Author

**A.S. Siva Suresh Kumar**  
Enterprise AI Architect | AI Transformation Lead  
20+ years in AI/ML | Former Head of AI, TVS Motor Company  
[LinkedIn](https://linkedin.com/in/siva-suresh-kumar-97725b78) · [GitHub](https://github.com/atkuri-siva-suresh)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
