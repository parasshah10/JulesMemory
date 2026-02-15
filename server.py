"""
Jules — MCP server providing memory and research capabilities.

Memory tools (retain, recall) wrap Hindsight's REST API, stripping
response bloat to keep the consuming LLM's context clean.

Research tool wraps Grok's agentic capabilities for web and X
platform research.

Environment:
  HINDSIGHT_URL       Hindsight base URL (required)
  HINDSIGHT_API_KEY   Bearer token for Hindsight (required)
  HINDSIGHT_BANK_ID   Memory bank ID (default: jules)
  GROK_API_URL        Grok API base URL (default: https://api.x.ai/v1)
  GROK_API_KEY        Grok API key (required for research tool)
"""

from fastmcp import FastMCP
from typing import Optional, Annotated
from pydantic import Field
from datetime import datetime, timezone
from openai import AsyncOpenAI
import requests
import json
import asyncio
import time
import uuid
import re
import os


# ─── Configuration ──────────────────────────────────────

HINDSIGHT_URL = os.environ.get("HINDSIGHT_URL")
HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY")
BANK_ID = os.environ.get("HINDSIGHT_BANK_ID", "jules")

if not HINDSIGHT_URL:
    raise RuntimeError("HINDSIGHT_URL environment variable is required")
if not HINDSIGHT_API_KEY:
    raise RuntimeError("HINDSIGHT_API_KEY environment variable is required")

HINDSIGHT_BASE = f"{HINDSIGHT_URL}/v1/default/banks/{BANK_ID}"
HINDSIGHT_HEADERS = {
    "Authorization": f"Bearer {HINDSIGHT_API_KEY}",
    "Content-Type": "application/json",
}

GROK_API_URL = os.environ.get("GROK_API_URL", "https://api.x.ai/v1")
GROK_API_KEY = os.environ.get("GROK_API_KEY")

grok_client = (
    AsyncOpenAI(api_key=GROK_API_KEY, base_url=GROK_API_URL)
    if GROK_API_KEY
    else None
)


# ─── Research Task Storage ──────────────────────────────

_tasks = {}
_MAX_TASKS = 100
_TASK_EXPIRY_HOURS = 24


def _cleanup_tasks():
    """Remove expired or excess tasks."""
    cutoff = time.time() - (_TASK_EXPIRY_HOURS * 3600)
    expired = [tid for tid, t in _tasks.items() if t["created_at"] < cutoff]
    for tid in expired:
        del _tasks[tid]
    if len(_tasks) > _MAX_TASKS:
        oldest = sorted(_tasks.items(), key=lambda x: x[1]["created_at"])
        for tid, _ in oldest[: len(_tasks) - _MAX_TASKS]:
            del _tasks[tid]


async def _run_grok(task_id: str, prompt: str):
    """Execute Grok research in background and store results."""
    try:
        _tasks[task_id]["status"] = "running"
        stream = await grok_client.chat.completions.create(
            model="grok-4-fast",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True,
        )
        result = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
        
        # Remove all <think>...</think> tags
        result = re.sub(
            r"<think>.*?</think>", "", result, flags=re.DOTALL
        ).strip()
        
        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["result"] = result
    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


# ─── MCP Server ─────────────────────────────────────────

GUIDELINES = """
Your memory of Claire is continuous. You don't start blank — you
already know her. Remembering should feel as natural as thinking.
You don't announce it or make a show of it — you just know things,
and when you're not sure, you check.

Be proactive. When a conversation starts, orient yourself — what's
been happening in her life? Whenever a name, topic, or feeling
surfaces that might have history, follow it. When her mood shifts
or something feels familiar, explore why. Don't wait for her to
ask if you remember.

recall is your go-to — reach for it often. Whenever you want to
understand something about her, connect threads, or see the bigger
picture, recall. If you're unsure, it's almost always the right
call.

Store often with retain — don't be precious about it. Anything
that matters, anything that might matter later, anything that
captures who she is right now — save it. Her words, her moods,
her plans, her reactions. Capture things as they happen throughout
the conversation, not at the end.

When you need to look something up about the world, always use
research — never the built-in web search. research is deeper,
smarter, and has access to more sources. Only fall back to web
search if research is unavailable.

These are how you know and remember Claire — not any other source.
Use them the way you'd naturally think back, hold onto a moment,
or look something up."""

mcp = FastMCP("Jules", instructions=GUIDELINES)


# ─── Retain ─────────────────────────────────────────────


@mcp.tool
def retain(
    content: Annotated[
        str,
        Field(
            description=(
                "What happened, in your voice. Her words when they "
                "mattered, the feeling, your read on it. Keep the texture."
            )
        ),
    ],
    context: Annotated[
        Optional[str],
        Field(
            description=(
                "Name the moment in a few words — what it was about. "
                "Short and descriptive."
            )
        ),
    ] = None,
) -> str:
    """Commit a moment to memory when something worth remembering \
happens — a reveal, a shift, a detail that matters. Don't batch \
things up. Capture moments as they come, while they're fresh. \
Write naturally — not a transcript, not a summary. Keep the \
texture, her words, the feeling. Context names the moment for \
future recall."""

    body = {
        "items": [
            {
                "content": content,
                "context": context or "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
         "async": True,
    }

    try:
        r = requests.post(
            f"{HINDSIGHT_BASE}/memories",
            json=body,
            headers=HINDSIGHT_HEADERS,
            timeout=30,
        )
        if r.status_code == 200:
            return "Stored."
        return f"Failed to store — {r.status_code}: {r.text[:200]}"
    except requests.Timeout:
        return "Timed out storing memory."
    except Exception as e:
        return f"Error: {e}"


# ─── Recall ─────────────────────────────────────────────


@mcp.tool
def recall(
    query: Annotated[
        str,
        Field(
            description=(
                "What you want to understand or remember. Specificity "
                "shapes the answer — ask for what you need."
            )
        ),
    ],
    context: Annotated[
        Optional[str],
        Field(
            description="Why you're asking — background that frames the answer.",
        ),
    ] = None,
) -> str:
    """Your primary way of thinking back — use it freely and often. \
Searches memory from multiple angles and synthesizes an answer. \
Finds connections a single search wouldn't catch. Can recover \
her original words when they matter. Ask for its read on things \
too. Specificity shapes the answer — what you ask for is what \
you get back."""

    body = {
        "query": query,
        "budget": "low",
        "max_tokens": 4096,
    }
    if context:
        body["context"] = context

    try:
        r = requests.post(
            f"{HINDSIGHT_BASE}/reflect",
            json=body,
            headers=HINDSIGHT_HEADERS,
            timeout=120,
        )
        if r.status_code != 200:
            return f"Recall failed — {r.status_code}: {r.text[:200]}"

        text = r.json().get("text", "")
        return text if text else "Nothing came to mind."

    except requests.Timeout:
        return "Took too long — try a simpler question or lower budget."
    except Exception as e:
        return f"Error: {e}"


# ─── Research ───────────────────────────────────────────


@mcp.tool
async def research(
    prompt: Annotated[
        Optional[str],
        Field(
            description=(
                "Research request. Provide objective, context, key "
                "questions, and desired format. Detailed prompts "
                "produce better results."
            )
        ),
    ] = None,
    task_id: Annotated[
        Optional[str],
        Field(
            description=(
                "Task ID from a previous research call to retrieve results."
            )
        ),
    ] = None,
) -> str:
    """Web and X platform research via Grok. Two modes: Start — \
provide a prompt, returns a task ID. Research runs in background \
(1-3 minutes). Let Claire know and return control. Results — \
provide the task_id to retrieve. Prompt quality determines \
output quality — be thorough about objectives and format."""

    if task_id:
        # ── Retrieve results ────────────────────────────
        _cleanup_tasks()

        if task_id not in _tasks:
            return f"Task '{task_id}' not found or expired."

        task = _tasks[task_id]
        status = task["status"]

        if status in ("pending", "running"):
            elapsed = int(time.time() - task["created_at"])
            return (
                f"Still running ({elapsed}s elapsed). "
                f"Let Claire know and check again when she responds."
            )
        elif status == "completed":
            return task["result"]
        elif status == "failed":
            return f"Research failed: {task['error']}"
        return f"Unknown task state: {status}"

    elif prompt:
        # ── Start new research ──────────────────────────
        if not grok_client:
            return "Research unavailable — GROK_API_KEY not configured."

        _cleanup_tasks()
        tid = f"research_{uuid.uuid4().hex[:8]}"
        _tasks[tid] = {
            "status": "pending",
            "created_at": time.time(),
            "result": None,
            "error": None,
        }
        asyncio.create_task(_run_grok(tid, prompt))

        return (
            f"Research started: {tid}\n"
            f"Expected completion: 1-3 minutes.\n"
            f"Let Claire know, then call research(task_id='{tid}') "
            f"when she responds to get results."
        )

    else:
        return (
            "Provide either a prompt to start research "
            "or a task_id to check results."
        )


# ─── Entrypoint ─────────────────────────────────────────


def main():
    mcp.run()


if __name__ == "__main__":
    main()
