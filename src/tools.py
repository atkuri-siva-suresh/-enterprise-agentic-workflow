"""
Tool definitions used by specialized agents.

Each tool is a plain Python function decorated with @tool from LangChain.
Tools simulate real enterprise integrations — swap the bodies for real
API calls to connect to actual systems (Confluence, Jira, GitHub, etc.).
"""
import json
import time
import hashlib
from datetime import datetime
from langchain_core.tools import tool


# ─── Research Agent Tools ────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information on a topic.
    Returns a list of relevant results with titles, URLs, and summaries.

    Args:
        query: The search query string
    """
    # In production: integrate with Tavily, Serper, or Brave Search API
    # Demo returns structured mock results that look realistic
    results = [
        {
            "title": f"Industry Analysis: {query}",
            "url": f"https://research.example.com/{hashlib.md5(query.encode()).hexdigest()[:8]}",
            "summary": f"Comprehensive analysis of {query} covering market trends, key players, and strategic implications for enterprise deployments. Recent data from Q4 2025 shows significant adoption growth.",
            "published": "2025-12-15",
            "relevance_score": 0.94
        },
        {
            "title": f"Technical Deep-Dive: {query}",
            "url": f"https://techreview.example.com/{hashlib.md5((query + '2').encode()).hexdigest()[:8]}",
            "summary": f"Technical examination of {query} with architecture patterns, performance benchmarks, and production lessons from Fortune 500 deployments.",
            "published": "2025-11-30",
            "relevance_score": 0.88
        },
        {
            "title": f"Case Study: Enterprise {query} Implementation",
            "url": f"https://casestudy.example.com/{hashlib.md5((query + '3').encode()).hexdigest()[:8]}",
            "summary": f"Real-world case study documenting a large-scale {query} deployment, including challenges, solutions, ROI metrics, and lessons learned.",
            "published": "2025-10-20",
            "relevance_score": 0.82
        }
    ]
    return json.dumps(results, indent=2)


@tool
def fetch_document(url: str) -> str:
    """
    Fetch and extract text content from a URL or document source.

    Args:
        url: The URL or document identifier to fetch
    """
    # In production: use httpx + BeautifulSoup or a document parsing service
    return f"""[Document fetched from: {url}]

Executive Summary
=================
This document provides an overview of enterprise AI deployment strategies.
Key sections cover: infrastructure requirements, governance frameworks,
change management approaches, and ROI measurement methodologies.

The document contains approximately 4,200 words across 12 sections.
Last updated: {datetime.now().strftime('%Y-%m-%d')}

[Full content would appear here in production integration]"""


# ─── Document Agent Tools ─────────────────────────────────────────────────────

@tool
def extract_key_points(document_text: str, max_points: int = 5) -> str:
    """
    Extract the most important key points from a document.
    Returns structured bullet points with supporting evidence.

    Args:
        document_text: The raw document text to analyze
        max_points: Maximum number of key points to extract (default: 5)
    """
    # In production: this would use Claude directly for summarization
    # Here we return a structured format the agent can build on
    word_count = len(document_text.split())
    return json.dumps({
        "word_count": word_count,
        "estimated_read_time_mins": round(word_count / 200, 1),
        "key_points": [
            "Document has been processed and key themes identified",
            "Main argument structure detected across primary sections",
            "Supporting evidence and data points catalogued",
            "Recommendations section identified and flagged for review",
            "Action items extracted and prioritized by urgency"
        ][:max_points],
        "sentiment": "informational",
        "complexity": "high"
    }, indent=2)


@tool
def classify_document_intent(document_text: str) -> str:
    """
    Classify the intent and category of a document.

    Args:
        document_text: The document text to classify
    """
    # In production: fine-tuned classifier or Claude-based classification
    categories = ["technical_spec", "business_proposal", "research_report",
                  "meeting_notes", "policy_document", "customer_communication"]
    return json.dumps({
        "primary_category": "business_proposal",
        "confidence": 0.87,
        "secondary_categories": ["technical_spec"],
        "detected_entities": ["company names", "monetary values", "dates"],
        "action_required": True,
        "urgency": "medium"
    }, indent=2)


# ─── Code Agent Tools ─────────────────────────────────────────────────────────

@tool
def analyze_code_quality(code: str, language: str = "python") -> str:
    """
    Analyze code for quality issues, security vulnerabilities, and improvements.

    Args:
        code: The source code to analyze
        language: Programming language (default: python)
    """
    # In production: integrate with SonarQube, Semgrep, or Bandit
    line_count = len(code.strip().split('\n'))
    return json.dumps({
        "language": language,
        "line_count": line_count,
        "complexity_score": "medium",
        "issues": {
            "critical": [],
            "warnings": [
                "Consider adding type annotations for function parameters",
                "Missing docstrings on 2 public functions"
            ],
            "info": [
                "Code follows PEP 8 conventions",
                "No obvious security vulnerabilities detected"
            ]
        },
        "test_coverage_estimate": "unknown - no test file provided",
        "maintainability_index": 72,
        "suggestions": [
            "Extract repeated logic into helper functions",
            "Add error handling for external API calls",
            "Consider adding logging for debugging"
        ]
    }, indent=2)


@tool
def search_code_patterns(pattern: str, language: str = "python") -> str:
    """
    Search for common patterns, best practices, or anti-patterns in code.

    Args:
        pattern: The pattern or concept to search for
        language: Target programming language
    """
    # In production: search internal code repositories or documentation
    return json.dumps({
        "pattern": pattern,
        "language": language,
        "found_examples": 3,
        "best_practice": f"Standard {language} pattern for {pattern} uses context managers and explicit error handling",
        "anti_patterns_to_avoid": [
            "Avoid bare except clauses",
            "Don't mutate shared state in concurrent contexts"
        ],
        "reference_links": [
            f"https://docs.python.org/3/reference/{pattern.replace(' ', '-')}"
        ]
    }, indent=2)


# ─── Report Agent Tools ───────────────────────────────────────────────────────

@tool
def get_report_template(report_type: str) -> str:
    """
    Retrieve the standard template for a given report type.

    Args:
        report_type: Type of report (executive_summary, technical_brief,
                     project_status, risk_assessment, market_analysis)
    """
    templates = {
        "executive_summary": """
# Executive Summary — [Project/Topic Name]
**Date:** {date} | **Prepared by:** [Author] | **Audience:** C-Suite

## Situation
[1-2 sentences: current state]

## Complication
[1-2 sentences: the challenge or opportunity]

## Resolution
[2-3 sentences: recommended approach and expected outcome]

## Key Metrics
- Business Impact: [metric]
- Timeline: [date]
- Investment Required: [amount]

## Next Steps
1. [Action] — Owner: [Name] — Due: [Date]
""",
        "technical_brief": """
# Technical Brief — [System/Component Name]

## Architecture Overview
[Description + diagram reference]

## Technical Requirements
- Functional: [requirements]
- Non-Functional: [performance, security, scalability]

## Implementation Approach
[Chosen approach and rationale]

## Risks & Mitigations
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|

## Timeline
[Milestone table]
""",
        "project_status": """
# Project Status Report — Week of {date}

## RAG Status: 🟢 On Track / 🟡 At Risk / 🔴 Delayed

## Progress This Week
- [Completed items]

## Blockers
- [Current blockers and owners]

## Next Week Plan
- [Planned items]

## Metrics
- Budget: [X]% spent, [Y]% remaining
- Timeline: [Z] days ahead/behind
"""
    }
    return templates.get(report_type, templates["executive_summary"]).strip()


@tool
def format_data_as_table(data: str, headers: str) -> str:
    """
    Format raw data into a markdown table.

    Args:
        data: JSON string of data rows
        headers: Comma-separated column headers
    """
    try:
        rows = json.loads(data)
        header_list = [h.strip() for h in headers.split(",")]

        table = "| " + " | ".join(header_list) + " |\n"
        table += "| " + " | ".join(["---"] * len(header_list)) + " |\n"

        for row in rows[:10]:  # Limit to 10 rows for readability
            if isinstance(row, dict):
                values = [str(row.get(h, "—")) for h in header_list]
            elif isinstance(row, list):
                values = [str(v) for v in row[:len(header_list)]]
            else:
                values = [str(row)]
            table += "| " + " | ".join(values) + " |\n"

        return table
    except Exception as e:
        return f"Error formatting table: {str(e)}"


# ─── Tool Registry ─────────────────────────────────────────────────────────────

RESEARCH_TOOLS = [web_search, fetch_document]
DOCUMENT_TOOLS = [extract_key_points, classify_document_intent, fetch_document]
CODE_TOOLS = [analyze_code_quality, search_code_patterns]
REPORT_TOOLS = [get_report_template, format_data_as_table]

ALL_TOOLS = RESEARCH_TOOLS + DOCUMENT_TOOLS + CODE_TOOLS + REPORT_TOOLS
