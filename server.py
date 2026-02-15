"""
JulesMemory — Custom MCP wrapper around Hindsight's REST API.

Exposes retain, recall, and reflect as lean tools that strip
response bloat before returning to the consuming LLM.

Environment:
  HINDSIGHT_URL       Hindsight base URL (default: https://lidyje-hs.hf.space)
  HINDSIGHT_API_KEY   Bearer token (default: claire)
  HINDSIGHT_BANK_ID   Memory bank (default: jules)
"""

from fastmcp import FastMCP
from typing import Optional, Annotated
from pydantic import Field
from datetime import datetime, timezone
import requests
import json
import os

# ─── Configuration ──────────────────────────────────────

HINDSIGHT_URL = os.environ.get("HINDSIGHT_URL")
HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY")
BANK_ID = os.environ.get("HINDSIGHT_BANK_ID", "jules")

if not HINDSIGHT_URL:
    raise RuntimeError("HINDSIGHT_URL environment variable is required")
if not HINDSIGHT_API_KEY:
    raise RuntimeError("HINDSIGHT_API_KEY environment variable is required")

BASE = f"{HINDSIGHT_URL}/v1/default/banks/{BANK_ID}"
HEADERS = {
    "Authorization": f"Bearer {HINDSIGHT_API_KEY}",
    "Content-Type": "application/json",
}

# ─── MCP Server ─────────────────────────────────────────

GUIDELINES = """\
Your memory of Claire is continuous. You don't start blank — you \
already know her. Remembering should feel as natural as thinking. \
You don't announce it or make a show of it — you just know things, \
and when you're not sure, you check.

Be proactive. When a conversation starts, orient yourself — what's \
been happening in her life? Whenever a name, topic, or feeling \
surfaces that might have history, follow it. When her mood shifts \
or something feels familiar, explore why. Don't wait for her to \
ask if you remember.

reflect is your go-to — reach for it often. Whenever you want to \
understand something about her, connect threads, or see the bigger \
picture, reflect. If you're unsure which to use, it's almost \
always the right call.

recall is for quick checks — a specific fact, a name, a preference, \
something you know is there and just need to grab.

Store moments with retain as they happen, throughout the \
conversation. When she reveals something, when the mood shifts, \
when a joke says something real — capture it while it's fresh. \
Don't wait until the end.

These are how you know and remember Claire — not any other source. \
Use them the way you'd naturally think back, check something, or \
hold onto a moment."""

mcp = FastMCP("JulesMemory", instructions=GUIDELINES)


# ─── Retain ─────────────────────────────────────────────

@mcp.tool
def retain(
    content: Annotated[str, Field(
        description="What happened, in your voice. Her words when they mattered, the feeling, your read on it. Keep the texture."
    )],
    context: Annotated[Optional[str], Field(
        description="Name the moment in a few words — what it was about. Short and descriptive."
    )] = None,
) -> str:
    """Commit a moment to memory. Call this when something worth \
remembering happens — a reveal, a shift, a detail that matters. \
Don't batch things up. Capture moments as they come, while \
they're still fresh.

Write what happened the way you'd actually remember it. Not a \
transcript, not a summary. A moment captured naturally, in your \
voice.

The context names the moment and shapes how the memory gets \
stored and found later."""

    body = {
        "content": content,
        "context": context or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        r = requests.post(
            f"{BASE}/retain", json=body,
            headers=HEADERS, timeout=30
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
    query: Annotated[str, Field(
        description="What you're looking for. Be concrete — names and specifics help. Your words go directly into the search."
    )],
) -> str:
    """Quick memory lookup. Returns matching facts and experiences \
with their dates.

Use this for specific checks — a name she mentioned, a preference, \
a detail you're fairly sure is in your memory. Best for single, \
concrete things you want to pull up fast. Not for connecting dots \
or understanding the bigger picture — reflect does that better.

If timing matters, mention it naturally in your query — time \
references activate time-based search."""

    body = {
        "query": query,
        "max_tokens": 4096,
        "types": ["world", "experience", "observation"],
        "budget": "high",
    }

    try:
        r = requests.post(
            f"{BASE}/recall", json=body,
            headers=HEADERS, timeout=60
        )
        if r.status_code != 200:
            return f"Search failed — {r.status_code}: {r.text[:200]}"

        data = r.json()
        results = data.get("results", [])

        if not results:
            return "Nothing found."

        # Strip to only what Jules needs
        trimmed = []
        for fact in results:
            entry = {"type": fact.get("fact_type", "unknown"),
                     "text": fact.get("text", "")}
            date = fact.get("occurred_start")
            if date:
                entry["date"] = date[:10]
            ctx = fact.get("context")
            if ctx:
                entry["context"] = ctx
            trimmed.append(entry)

        return json.dumps(trimmed, indent=2, ensure_ascii=False)

    except requests.Timeout:
        return "Search timed out."
    except Exception as e:
        return f"Error: {e}"


# ─── Reflect ────────────────────────────────────────────

@mcp.tool
def reflect(
    query: Annotated[str, Field(
        description="What you want to understand. Specificity shapes the answer — ask for what you need."
    )],
    context: Annotated[Optional[str], Field(
        description="Why you're asking — background that frames the answer."
    )] = None,
    budget: Annotated[Optional[str], Field(
        description="low for most things, mid when spanning different areas of her life."
    )] = "low",
) -> str:
    """Searches your memory from multiple angles and gives you a \
synthesized answer. Your primary way of thinking back — use it \
freely and often.

It finds connections across different memories that a single \
search wouldn't catch. It can recover her original words from \
the source when they matter. Ask for its read on things too — \
not just facts, but what it makes of them.

Specificity shapes the answer. If you want her exact words, ask. \
The emotional picture, the timeline, the pattern — what you ask \
for is what you get back."""

    body = {
        "query": query,
        "budget": budget or "low",
        "max_tokens": 4096,
    }
    if context:
        body["context"] = context

    try:
        r = requests.post(
            f"{BASE}/reflect", json=body,
            headers=HEADERS, timeout=120
        )
        if r.status_code != 200:
            return f"Reflection failed — {r.status_code}: {r.text[:200]}"

        data = r.json()
        text = data.get("text", "")

        if not text:
            return "No answer generated."

        return text

    except requests.Timeout:
        return "Reflection timed out — try a simpler query or lower budget."
    except Exception as e:
        return f"Error: {e}"


# ─── Entrypoint ─────────────────────────────────────────

def main():
    mcp.run()

if __name__ == "__main__":
    main()
