"""
Demo script for enterprise-agentic-workflow.

Run this to verify your setup is working correctly.
Each example demonstrates a different agent path.

Usage:
    python examples/demo.py
    python examples/demo.py --task research
    python examples/demo.py --task code
"""
import sys
import os
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.workflow import run_workflow
from src.config import config

console = Console()

DEMO_TASKS = {
    "research": (
        "Research the current state of agentic AI frameworks in 2025. "
        "Focus on LangGraph vs AutoGen vs CrewAI comparison for enterprise deployments."
    ),
    "document": (
        "Analyze the following policy document excerpt and identify any compliance risks:\n"
        "EXCERPT: 'All customer data must be retained for 3 years. "
        "Access logs are optional but recommended. "
        "Encryption at rest is required for PII fields only.'"
    ),
    "code": (
        "Review this Python code for security issues and improvements:\n"
        "```python\n"
        "import os\n"
        "def load_config(path):\n"
        "    with open(path) as f:\n"
        "        return eval(f.read())  # Load config file\n"
        "```"
    ),
    "report": (
        "Generate an executive summary report for the following AI initiative:\n"
        "Project: 'Enterprise LLM Platform'\n"
        "Status: On track\n"
        "Q3 Progress: 3 of 5 milestones completed\n"
        "Budget: Rs.45L of Rs.60L spent (75%)\n"
        "Key achievement: Reduced manual reporting by 70%\n"
        "Risk: 1 integration dependency delayed by vendor"
    ),
    "approval": (
        "Send a follow-up email to all enterprise clients announcing "
        "the new AI platform launch scheduled for next Monday."
        # Note: This will trigger requires_approval=True
    ),
}


def print_trace(trace: list) -> None:
    """Print agent execution trace as a Rich table."""
    table = Table(title="Agent Execution Trace", show_header=True, header_style="bold cyan")
    table.add_column("Step", style="dim", width=4)
    table.add_column("Agent", style="bold")
    table.add_column("Action", style="yellow")
    table.add_column("Duration", justify="right", style="green")
    table.add_column("Output Preview", style="dim")

    for i, entry in enumerate(trace, 1):
        table.add_row(
            str(i),
            entry.get("agent", "?").replace("_", " ").title(),
            entry.get("action", "?"),
            f"{entry.get('duration_ms', 0)}ms",
            entry.get("output_summary", "")[:60] + "...",
        )

    console.print(table)


def run_demo(task_name: str) -> None:
    """Run a single demo task and display results."""
    task = DEMO_TASKS.get(task_name)
    if not task:
        console.print(f"[red]Unknown task: {task_name}. Choose from: {list(DEMO_TASKS.keys())}[/red]")
        return

    console.print(Panel(
        f"[bold cyan]Task:[/bold cyan] {task[:100]}...\n"
        f"[bold green]Expected Agent:[/bold green] {task_name}_agent",
        title=f"Running: {task_name.title()} Example",
        expand=False,
    ))

    try:
        result = run_workflow(
            user_message=task,
            thread_id=f"demo-{task_name}",
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    console.print(f"\n[bold]Task Type:[/bold] {result['task_type']}")
    console.print(f"[bold]Requires Approval:[/bold] {result['requires_approval']}")
    console.print(f"[bold]Agents in Trace:[/bold] {len(result['trace'])}")
    console.print(f"[bold]~Tokens Used:[/bold] {result['total_tokens_used']:,}\n")

    print_trace(result["trace"])

    console.print("\n")
    console.print(Panel(
        Markdown(result["final_response"]),
        title="Final Response",
        border_style="green",
    ))


def run_all_demos() -> None:
    """Run all demo tasks sequentially."""
    for task_name in DEMO_TASKS.keys():
        run_demo(task_name)
        console.print("\n" + "─" * 80 + "\n")


if __name__ == "__main__":
    # Validate config first
    try:
        config.validate()
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("[yellow]Create a .env file from .env.example and add your ANTHROPIC_API_KEY[/yellow]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Enterprise Agentic Workflow Demo")
    parser.add_argument(
        "--task",
        choices=list(DEMO_TASKS.keys()) + ["all"],
        default="research",
        help="Which demo task to run (default: research)",
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Enterprise Agentic Workflow[/bold]\n"
        "Multi-agent AI orchestration with LangGraph + Claude\n\n"
        f"Model: [cyan]{config.MODEL_NAME}[/cyan]\n"
        f"LangSmith: [cyan]{'Enabled' if config.is_langsmith_enabled() else 'Disabled'}[/cyan]",
        title="Setup",
        border_style="blue",
    ))

    if args.task == "all":
        run_all_demos()
    else:
        run_demo(args.task)
