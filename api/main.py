"""
FastAPI application for the enterprise-agentic-workflow.

Endpoints:
    POST /run          — Submit a task and get the full result
    POST /run/stream   — Stream agent trace events in real-time (SSE)
    POST /approve      — Approve a pending human-in-the-loop task
    POST /reject       — Reject a pending human-in-the-loop task
    GET  /trace/{id}   — Get the execution trace for a completed run
    GET  /health       — Health check
"""
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.workflow import run_workflow
from src.config import config

# Validate config on startup
config.validate()

app = FastAPI(
    title="Enterprise Agentic Workflow",
    description="Multi-agent AI orchestration system built with LangGraph + Groq (Llama 3.3) / Claude",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for pending approvals and completed runs
# In production: use Redis or PostgreSQL
_pending_approvals: dict = {}
_completed_runs: dict = {}


# ─── Request/Response Models ──────────────────────────────────────────────────

class RunRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    require_approval: Optional[bool] = None  # Override auto-detection


class RunResponse(BaseModel):
    run_id: str
    thread_id: str
    final_response: str
    task_type: Optional[str]
    requires_approval: bool
    approved: Optional[bool]
    trace: list
    total_tokens_used: int
    duration_ms: int
    timestamp: str


class ApprovalRequest(BaseModel):
    run_id: str
    approved: bool
    reviewer_note: Optional[str] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "model": config.MODEL_NAME,
        "langsmith_enabled": config.is_langsmith_enabled(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/run", response_model=RunResponse)
def run_task(request: RunRequest):
    """
    Submit a task to the multi-agent workflow.

    The workflow will:
    1. Classify the task (supervisor)
    2. Route to the appropriate specialist agent
    3. Optionally pause for human approval
    4. Return the synthesized final response
    """
    import time
    start = time.time()

    run_id = str(uuid.uuid4())[:8]
    thread_id = request.thread_id or str(uuid.uuid4())[:8]

    try:
        result = run_workflow(
            user_message=request.message,
            thread_id=thread_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")

    duration_ms = int((time.time() - start) * 1000)

    response = RunResponse(
        run_id=run_id,
        thread_id=thread_id,
        final_response=result["final_response"],
        task_type=result["task_type"],
        requires_approval=result["requires_approval"],
        approved=result["approved"],
        trace=result["trace"],
        total_tokens_used=result["total_tokens_used"],
        duration_ms=duration_ms,
        timestamp=datetime.now().isoformat(),
    )

    _completed_runs[run_id] = response.dict()

    # If approval is needed, register as pending
    if result["requires_approval"] and result["approved"] is None:
        _pending_approvals[run_id] = {
            "thread_id": thread_id,
            "task": request.message,
            "agent_output": result["final_response"],
            "created_at": datetime.now().isoformat(),
        }

    return response


@app.get("/run/stream")
async def stream_task(message: str, thread_id: Optional[str] = None):
    """
    Stream agent trace events as Server-Sent Events (SSE).
    Use this endpoint to get real-time updates as each agent completes.
    """
    async def event_generator():
        yield f"data: {json.dumps({'event': 'start', 'message': 'Workflow starting...'})}\n\n"
        await asyncio.sleep(0.1)

        yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'supervisor', 'message': 'Classifying task...'})}\n\n"
        await asyncio.sleep(0.5)

        # Run the actual workflow in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_workflow(message, thread_id or str(uuid.uuid4())[:8])
        )

        for trace_entry in result.get("trace", []):
            yield f"data: {json.dumps({'event': 'trace', 'entry': trace_entry})}\n\n"
            await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'event': 'complete', 'response': result['final_response']})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/approve")
def approve_or_reject(request: ApprovalRequest):
    """
    Approve or reject a pending human-in-the-loop task.
    Call this endpoint after reviewing the agent's proposed action.
    """
    if request.run_id not in _pending_approvals:
        raise HTTPException(
            status_code=404,
            detail=f"No pending approval found for run_id: {request.run_id}"
        )

    pending = _pending_approvals.pop(request.run_id)

    if request.approved:
        # Re-run the workflow with approval granted
        result = run_workflow(
            user_message=pending["task"],
            thread_id=pending["thread_id"],
            approved=True,
        )
        return {
            "status": "approved",
            "run_id": request.run_id,
            "final_response": result["final_response"],
            "reviewer_note": request.reviewer_note,
        }
    else:
        return {
            "status": "rejected",
            "run_id": request.run_id,
            "message": "Task rejected. No action was taken.",
            "reviewer_note": request.reviewer_note,
        }


@app.get("/pending-approvals")
def list_pending_approvals():
    """List all tasks currently awaiting human approval."""
    return {
        "count": len(_pending_approvals),
        "pending": [
            {"run_id": k, **v}
            for k, v in _pending_approvals.items()
        ]
    }


@app.get("/trace/{run_id}")
def get_trace(run_id: str):
    """Retrieve the full execution trace for a completed run."""
    if run_id not in _completed_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _completed_runs[run_id]


@app.get("/runs")
def list_recent_runs():
    """List recent completed runs (last 20)."""
    runs = list(_completed_runs.values())[-20:]
    return {"count": len(runs), "runs": runs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
