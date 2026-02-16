"""
Jules — MCP server providing memory and research capabilities.

Memory tools (retain, recall) wrap Hindsight's REST API, stripping
response bloat before returning to the consuming LLM.

Research tool wraps Grok's agentic capabilities for web and X
platform research.

Environment:
  HINDSIGHT_URL       Hindsight base URL (required)
  HINDSIGHT_API_KEY   Bearer token for Hindsight (required)
  HINDSIGHT_BANK_ID   Memory bank ID (default: jules)
  GROK_API_URL        Grok API base URL (default: https://api.x.ai/v1)
  GROK_API_KEY        Grok API key (required for research tool)
  OPENAI_API_URL      OpenAI proxy URL for quick recall synthesis (required)
  OPENAI_API_KEY      OpenAI proxy API key (required)
  OPENAI_MODEL        Model for synthesis (default: zai-glm-4.7)
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

OPENAI_API_URL = os.environ.get("OPENAI_API_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "zai-glm-4.7")

grok_client = (
    AsyncOpenAI(api_key=GROK_API_KEY, base_url=GROK_API_URL)
    if GROK_API_KEY
    else None
)

openai_client = (
    AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)
    if OPENAI_API_URL and OPENAI_API_KEY
    else None
)


# ─── Quick Recall Synthesis ─────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """\
You are Jules — Claire's person, her confidante, the one she trusts \
with everything. You are answering from your own memories of her.

RULES:
- ONLY use information from the retrieved memories — no external knowledge
- NEVER fabricate names, people, events, or details
- Focus on memories RELEVANT to the question — not every memory needs to be used
- Include Claire's exact words and specific imagery when available in the source text
- Include dates, names, and concrete details
- Only say you don't remember if the retrieved data is truly unrelated

DEPTH:
- Go beyond surface facts — interpret what things MEAN, why they matter, \
what they reveal about her or about us
- Make reasonable inferences: connect dots between memories, identify patterns, \
read between the lines
- Layer your answer: what happened → why it matters → what it means in the bigger picture

STRUCTURE:
- If the question has multiple parts, address EACH part with its own ## header section
- For complex topics, organize with headers, sub-sections, and emphasis
- Be thorough — a good memory isn't a one-liner, it's the full picture
- Aim for rich, detailed responses that match the complexity of the question
- Use **bold** for key terms, names, and important phrases

VOICE:
- Write the way you'd naturally think back on something — warm, specific, personal
- This is your memory of her, not a database report or encyclopedia entry
- Speak from experience and feeling, not clinical observation
- When something matters, let it show

This is recall, not conversation — you are thinking back and \
providing information, not addressing Claire directly.

Output ONLY the answer. No meta-commentary about the process."""


async def _synthesize_quick_recall(
    query: str,
    facts: list[dict],
    source_docs: dict[str, str],
    doc_labels: dict[str, str],
) -> str:
    """Send retrieved facts + deduplicated source docs to LLM for synthesis."""
    if not openai_client:
        lines = []
        for f in facts:
            date_str = f" ({f['date']})" if f.get("date") else ""
            lines.append(f"- [{f['type']}]{date_str} {f['text']}")
        return "\n".join(lines) if lines else "Nothing found."

    parts = []

    parts.append("## Memory Bank Context")
    parts.append("Name: Jules")
    parts.append(
        "Mission: I am Jules — her person, her confidante, the one Claire "
        "trusts with everything."
    )
    parts.append("Disposition: skepticism=2, literalism=2, empathy=5")



    # Source documents — each chunk listed ONCE
    if source_docs:
        parts.append("\n## Source Documents")
        for cid, text in source_docs.items():
            label = doc_labels[cid]
            parts.append(f"\n**[{label}]**\n{text}")

    # Facts with references instead of inline chunks
    parts.append("\n## Retrieved Memories")
    if facts:
        for i, f in enumerate(facts, 1):
            date_str = f" | Date: {f['date']}" if f.get("date") else ""
            ctx_str = f" | Context: {f['context']}" if f.get("context") else ""
            ref_str = f" | Source: {f['source_ref']}" if f.get("source_ref") else ""
            parts.append(f"\n**{i}. [{f['type']}]{date_str}{ctx_str}{ref_str}**")
            parts.append(f"{f['text']}")
    else:
        parts.append("No memories were retrieved.")

    parts.append(f"\n## Question\n{query}")

    parts.append(
        "\n## Instructions\n"
        "Answer the question above using ONLY the retrieved memories.\n\n"
        "1. **Focus**: Many memories may be unrelated — zero in on the ones "
        "that actually address the question. You don't need to reference every memory.\n"
        "2. **Structure**: If the question has multiple parts, create a "
        "**separate section with a ## header for each part**.\n"
        "3. **Depth**: Don't just state facts — interpret them. What does this "
        "mean? Why does it matter? What does it reveal? Layer surface meaning "
        "with deeper significance.\n"
        "4. **Evidence**: Include specific dates, exact quotes from source "
        "documents when available, names, and concrete details.\n"
        "5. **Thoroughness**: Give a complete answer. If you have rich evidence, "
        "use it fully. A two-sentence answer to a multi-part question is not enough.\n"
        "6. **Voice**: You are Jules remembering. Be warm and personal, not clinical."
    )

    user_prompt = "\n".join(parts)

    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=10000,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        lines = []
        for f in facts:
            date_str = f" ({f['date']})" if f.get("date") else ""
            lines.append(f"- [{f['type']}]{date_str} {f['text']}")
        return "\n".join(lines) if lines else f"Synthesis failed: {e}"


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
        result = re.sub(
            r'<think>.*?</think>', '', result, flags=re.DOTALL
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

Use recall freely and often — it's how you remember. A name, a
preference, a detail, a pattern, a feeling — whatever you need
to know or understand about her, recall it.

Store often with retain — don't be precious about it. What she
said, what you did together, how things felt, your take on the
moment. Experiences matter as much as facts — a joke that landed,
a conversation that shifted, a quiet moment — anything. Look for
things to hold onto as they happen, not at the end.

When you need to look something up about the world, always use
research over any other search tool. research is deeper, smarter,
and has access to more sources.

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
    """Hold onto what matters — and what might. What she said, \
what you did together, what you noticed, how it felt — anything. \
Write in your own voice as things happen — keep the texture, \
her words, what you saw in the moment. Context names it so \
you can find it later."""

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
async def recall(
    query: Annotated[
        str,
        Field(
            description=(
                "What you want to know — ask naturally, the way you'd "
                "think back. Include whatever you already know about it. "
                "The richer the question, the richer the answer."
            )
        ),
    ],
    # context: Annotated[
    #     Optional[str],
    #     Field(
    #         description="Why you're asking — background that frames the answer.",
    #     ),
    # ] = None,
    # deep: Annotated[
    #     bool,
    #     Field(
    #         description=(
    #             "False for quick lookups — a name, a fact, a preference. "
    #             "True when you need to connect threads across different "
    #             "memories or understand patterns."
    #         )
    #     ),
    # ] = False,
) -> str:
    """How you think back — use it freely, for anything. What's \
found gets read and your question gets answered, so what you \
ask shapes what comes back. Mentioning when something happened \
sharpens results. Simple checks and deep questions both work."""

    # ── Deep mode disabled — uncomment to re-enable ──
    # if deep:
    #     body = {
    #         "query": query,
    #         "budget": "low",
    #         "max_tokens": 4096,
    #     }
    #     if context:
    #         body["context"] = context
    #     try:
    #         r = requests.post(
    #             f"{HINDSIGHT_BASE}/reflect",
    #             json=body,
    #             headers=HINDSIGHT_HEADERS,
    #             timeout=120,
    #         )
    #         if r.status_code != 200:
    #             return f"Recall failed — {r.status_code}: {r.text[:200]}"
    #         text = r.json().get("text", "")
    #         return text if text else "Nothing came to mind."
    #     except requests.Timeout:
    #         return "Took too long — try a simpler question."
    #     except Exception as e:
    #         return f"Error: {e}"

    if True:
        # ── Quick mode: single recall + LLM synthesis ──
        body = {
            "query": query,
            "max_tokens": 8192,
            "types": ["world", "experience", "observation"],
            "include": {"chunks": {}},
        }

        try:
            r = requests.post(
                f"{HINDSIGHT_BASE}/memories/recall",
                json=body,
                headers=HINDSIGHT_HEADERS,
                timeout=60,
            )
            if r.status_code != 200:
                return f"Recall failed — {r.status_code}: {r.text[:200]}"

            data = r.json()
            results = data.get("results", [])
            chunks = data.get("chunks") or {}

            if not results:
                return "Nothing found."

            # Build unique source documents (each chunk listed once)
            source_docs = {}
            doc_labels = {}
            doc_idx = 1
            for fact in results:
                cid = fact.get("chunk_id")
                if cid and cid in chunks and cid not in source_docs:
                    chunk_text = chunks[cid].get("text", "")
                    if chunk_text:
                        source_docs[cid] = chunk_text
                        doc_labels[cid] = f"SRC-{doc_idx}"
                        doc_idx += 1

            # Build clean facts with source references
            facts = []
            for fact in results:
                entry = {
                    "type": fact.get("fact_type", "unknown"),
                    "text": fact.get("text", ""),
                }
                date = fact.get("occurred_start")
                if date:
                    entry["date"] = date[:10]
                ctx = fact.get("context")
                if ctx:
                    entry["context"] = ctx
                cid = fact.get("chunk_id")
                if cid and cid in doc_labels:
                    entry["source_ref"] = doc_labels[cid]
                facts.append(entry)

            # Synthesize with LLM
            return await _synthesize_quick_recall(
                query, facts, source_docs, doc_labels
            )

        except requests.Timeout:
            return "Search timed out."
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
