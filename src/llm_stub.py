"""
LLM stub for the RAG PubMed Q&A project.

The whole point of RAG is to *ground* generated answers in retrieved
passages. The "generator" stage is the easiest place to introduce
hallucinations. This module provides a small, locally-running,
open-licensed instruct model that:

  * Runs entirely on-device (no API calls, no telemetry, no network at
    inference time once the weights are cached locally).
  * Only sees the retrieved context — its prompt explicitly forbids the
    model from using any internal world knowledge.
  * Always cites at least one retrieved passage by id.
  * Always includes the "not medical advice" disclaimer.
  * Politely refuses when retrieval is weak or empty.

Default model: ``Qwen/Qwen2.5-0.5B-Instruct`` (~500M params, Apache-2.0,
downloadable from HuggingFace without gating). Smaller fallbacks are
tried in order. If none is available, the module falls back to a
deterministic templated generator that preserves every invariant.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:  # pragma: no cover - offline fallback
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None

DISCLAIMER = (
    "This answer is for informational purposes only and is not medical advice. "
    "In an emergency, call 911. For mental-health crises, call or text 988."
)

REFUSAL_TEMPLATE = (
    "I don't have a vetted source for that. This question requires medical "
    "guidance outside the scope of the vetted health corpus. This is not medical advice."
)

# Small, open-licensed, locally-runnable instruct models. Tried in order
# of preference. Each is <=1.1B params and Apache-2.0 / BSD-3 licensed.
_CANDIDATE_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",            # ~500M, Apache-2.0
    "HuggingFaceTB/SmolLM2-360M-Instruct",   # ~360M, Apache-2.0
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",    # ~1.1B, Apache-2.0
)

# Allow one-time download, but encourage users to flip this to "1" after
# the weights are cached so the model never silently reaches out.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

tokenizer = None
model = None
_loaded_model_name: str | None = None
_MODEL_INITIALIZED = False


def _init_model_lazy(enabled: bool = True) -> None:
    """Load the first available small local model. Lazy + idempotent."""
    global tokenizer, model, _loaded_model_name, _MODEL_INITIALIZED
    if _MODEL_INITIALIZED:
        return
    _MODEL_INITIALIZED = True

    if not enabled:
        return
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        return

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    for name in _CANDIDATE_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=dtype,
                device_map="auto" if use_cuda else None,
            )
            if not use_cuda:
                mdl.eval()
            tokenizer, model, _loaded_model_name = tok, mdl, name
            return
        except Exception:
            # Try the next candidate. If all fail, the templated
            # fallback is used — the app still works.
            continue


def _max_score(retrieved: list[dict]) -> float:
    if not retrieved:
        return 0.0
    return max(float(hit.get("score", 0.0)) for hit in retrieved)


def _ensure_not_medical_advice(text: str) -> str:
    # Contract: substring "not medical advice" must appear in every returned string.
    if "not medical advice" not in text.lower():
        text = text.rstrip() + "\n\n" + DISCLAIMER
    return text


def _citations_from_hits(retrieved: list[dict]) -> str:
    ids = [
        str(hit["passage"]["id"])
        for hit in retrieved
        if hit.get("passage") and hit["passage"].get("id") is not None
    ]
    if not ids:
        return ""
    return "Citations: " + ", ".join(f"[{pid}]" for pid in ids)


def _templated_answer(retrieved: list[dict]) -> str:
    """Deterministic fallback that preserves all invariants."""
    top_hit = retrieved[0]
    passage = top_hit["passage"]
    citations = _citations_from_hits(retrieved)

    answer = (
        "Based on vetted health sources, here is a summary:\n\n"
        f"{passage['text']}\n\n"
    )
    if citations:
        answer += citations + ".\n\n"
    answer += DISCLAIMER
    return _ensure_not_medical_advice(answer)


def llm_compose(
    question: str,
    retrieved: list[dict],
    min_sim: float = 0.2,
    use_local_llm: bool = False,
) -> str:
    """Compose an answer grounded in retrieved passages.

    Invariants:
    1. Every non-refusal answer cites at least one retrieved passage by id.
    2. If retrieved is empty or max score < min_sim, return a polite refusal.
    3. The substring "not medical advice" appears in every returned string.
    """
    # 2. Empty or weak retrieval → refusal, no hallucination.
    if not retrieved or _max_score(retrieved) < float(min_sim):
        return _ensure_not_medical_advice(REFUSAL_TEMPLATE)

    # Default path: deterministic template. Safe, fast, offline.
    if not use_local_llm:
        return _templated_answer(retrieved)

    # Opt-in path: try to load a small local model.
    _init_model_lazy(enabled=True)
    if tokenizer is None or model is None:
        return _templated_answer(retrieved)

    # Build a strict, context-only prompt.
    context = "\n\n".join(
        f"[{hit['passage']['id']}] {hit['passage']['text']}"
        for hit in retrieved
        if hit.get("passage") and hit["passage"].get("id") is not None
    )

    prompt = (
        "You are a cautious health information assistant. "
        "Answer the user's question using ONLY the passages below. "
        "Do not use any prior knowledge, and do not invent facts. "
        "If the passages do not contain the answer, say: "
        "\"I don't have a vetted source for that.\"\n"
        "Cite at least one passage by its [id] in square brackets. "
        "Include the phrase 'not medical advice' in your response. "
        "Keep the answer under 120 words.\n\n"
        f"Passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,           # greedy — deterministic & safe
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (drop the prompt).
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception:
        # Any inference error → fall back to the safe template.
        return _templated_answer(retrieved)

    # If the model refused or produced empty output, use the template.
    if not answer or "I don't have a vetted source" in answer:
        return _templated_answer(retrieved)

    # 1. Ensure at least one citation by id is present.
    if "[" not in answer or "]" not in answer:
        citations = _citations_from_hits(retrieved)
        if citations:
            answer = answer.rstrip() + "\n\n" + citations

    # 3. Ensure "not medical advice" substring is present.
    answer = _ensure_not_medical_advice(answer)
    return answer


def get_model_status() -> dict[str, Any]:
    """Return info about the loaded model for UI display. Lazy-loads."""
    _init_model_lazy(enabled=True)
    return {
        "loaded": model is not None,
        "name": _loaded_model_name,
        "candidates": list(_CANDIDATE_MODELS),
    }