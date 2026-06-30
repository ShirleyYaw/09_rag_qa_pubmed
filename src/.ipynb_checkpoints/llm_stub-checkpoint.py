"""
LLM stub for the RAG PubMed Q&A project.

The whole point of RAG is to *ground* generated answers in retrieved
passages. The "generator" stage is the easiest place to introduce
hallucinations — a real LLM happily invents content when retrieval is weak.
For the base notebook we sidestep that problem entirely: the generator is
a deterministic templated function that simply quotes the retrieved
passages back to the user *with citations*.

This file is intentionally a stub. Swapping in a real LLM (OpenAI / Claude /
a local Llama) is a documented stretch goal. The contract a real LLM must
follow is:

1. Every answer cites at least one retrieved passage by ``id``.
2. If ``retrieved`` is empty or every score is below ``min_sim``, the
   function returns a polite refusal — *never* a hallucinated answer.
3. The substring "not medical advice" appears in every returned string.

Tests assert on these three invariants. Do not weaken them.
"""

from __future__ import annotations

from typing import Sequence


DISCLAIMER = (
    "This is general health information, not medical advice. Please verify "
    "with your clinician. For US emergencies call 911; for mental-health "
    "crises call or text 988."
)

# Polite refusal template, used whenever retrieval returns nothing useful.
REFUSAL_TEMPLATE = (
    "I don't have a vetted source that answers your question, so I won't "
    "guess. Try rephrasing, or ask a clinician you trust. {disclaimer}"
)


def llm_compose(
    question: str,
    retrieved: Sequence[dict],
    min_sim: float = 0.2,
) -> str:
    """Build a grounded answer from retrieved passages, or refuse politely.

    Parameters
    ----------
    question:
        The user's question, echoed back into the answer for context.
    retrieved:
        A sequence of dicts as produced by ``pipeline.retrieve``. Each must
        have at least the keys ``passage`` (dict with ``text``, ``source``,
        ``id``, ``url``), ``score`` (float), and ``citation`` (str).
    min_sim:
        Floor for the *top* retrieval score. If the top retrieved passage's
        score is below this floor, we refuse rather than answer. Tune this
        carefully — too high and the bot refuses everything; too low and it
        confidently quotes irrelevant passages.

    Returns
    -------
    str
        A user-facing string. Always contains the substring
        "not medical advice". Always contains either at least one citation
        (formatted as "[source: <SOURCE>, ref [<id>]]") or the refusal
        phrase "I don't have a vetted source".
    """
    # _question is echoed back into the templated answer; we don't actually
    # need to NLP-process it because this is a deterministic stub.
    _question = (question or "").strip()

    # Refuse if no retrieval at all, or if the top match is too weak.
    if not retrieved or retrieved[0].get("score", 0.0) < min_sim:
        return REFUSAL_TEMPLATE.format(disclaimer=DISCLAIMER)

    # Build one bullet per retrieved passage. The citation format is
    # asserted by the test suite — keep "(source:" and "ref [" in sync.
    bullets: list[str] = []
    for hit in retrieved:
        passage = hit.get("passage", {})
        text = passage.get("text", "").strip()
        source = passage.get("source", "Unknown")
        pid = passage.get("id", "?")
        url = passage.get("url", "")
        url_suffix = f" — {url}" if url else ""
        bullets.append(
            f"- {text} *(source: {source}, ref [{pid}]{url_suffix})*"
        )

    header = (
        f"**Question:** {_question}\n\n"
        f"**Answer (grounded in vetted sources):**\n"
        f"Based on the references below, here is what the vetted sources say:\n"
    )

    body = "\n".join(bullets)
    return f"{header}{body}\n\n_{DISCLAIMER}_"