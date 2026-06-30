"""
RAG pipeline for the PubMed Q&A project.

The whole pipeline is three small functions:

- build_index(passages, embedder): embed every passage in the corpus.
- retrieve(question, index, embedder, top_k, min_sim): return relevant passages.
- compose_answer(question, retrieved): answer with citations or refuse.

The cite-or-refuse guardrail lives in two places:

1. retrieve filters out passages that do not clear min_sim.
2. retrieve also rejects broad TF-IDF matches with too little topic overlap.
3. llm_compose returns a refusal when retrieval is empty or weak.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .embedder import EmbeddingModel
from .llm_stub import DISCLAIMER, llm_compose


SEED = 42

_STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "before",
    "being", "benefit", "benefits", "between", "could", "does", "doing",
    "during", "from", "have", "health", "help", "helps", "into", "more",
    "most", "should", "than", "that", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "what", "when", "where",
    "which", "while", "with", "would", "your",
}


@dataclass
class Index:
    """Tiny in-memory vector store."""

    vectors: np.ndarray
    passages: list[dict[str, Any]]
    embedder_name: str

    @property
    def size(self) -> int:
        return len(self.passages)


def _meaningful_tokens(text: str) -> set[str]:
    """Return content-ish tokens for lightweight relevance checks."""
    words = re.findall(r"[a-z][a-z0-9-]{2,}", text.lower())
    return {word for word in words if word not in _STOPWORDS}


def _passage_text(passage: dict[str, Any]) -> str:
    """Combine a passage's title and text for embedding."""
    title = (passage.get("title") or "").strip()
    body = (passage.get("text") or "").strip()
    if title and body:
        return f"{title}. {body}"
    return title or body


def _has_enough_lexical_overlap(question: str, passage: dict[str, Any]) -> bool:
    """Reject broad TF-IDF neighbors that share no specific query terms."""
    question_tokens = _meaningful_tokens(question)
    if not question_tokens:
        return True

    passage_tokens = _meaningful_tokens(_passage_text(passage))
    overlap = question_tokens & passage_tokens

    if len(question_tokens) <= 2:
        return len(overlap) >= 1
    return len(overlap) >= 2


def build_index(
    passages: Sequence[dict[str, Any]],
    embedder: EmbeddingModel,
) -> Index:
    """Embed every passage once and return an Index."""
    np.random.seed(SEED)
    random.seed(SEED)

    if not passages:
        raise ValueError("build_index needs at least one passage.")

    texts = [_passage_text(p) for p in passages]
    vectors = embedder.embed(texts)

    return Index(
        vectors=np.asarray(vectors, dtype=np.float32),
        passages=list(passages),
        embedder_name=embedder.name,
    )


def _format_citation(passage: dict[str, Any]) -> str:
    """Pretty citation string used in UI rendering."""
    source = passage.get("source", "Unknown")
    pid = passage.get("id", "?")
    url = passage.get("url", "")
    if url:
        return f"[{source}, ref {pid}] {url}"
    return f"[{source}, ref {pid}]"


def retrieve(
    question: str,
    index: Index,
    embedder: EmbeddingModel,
    top_k: int = 3,
    min_sim: float = 0.2,
) -> list[dict[str, Any]]:
    """Return the top-k passages above the relevance floor."""
    if not question or not question.strip():
        return []
    if index.size == 0:
        return []

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        query_vec = embedder.embed([question])
        if query_vec.shape[0] == 0:
            return []

        sims = (
            index.vectors.astype(np.float64, copy=False)
            @ query_vec[0].astype(np.float64, copy=False)
        )
        sims = np.where(np.isfinite(sims), sims, -np.inf)

    order = np.argsort(-sims)
    hits: list[dict[str, Any]] = []

    for rank, i in enumerate(order):
        if len(hits) >= top_k:
            break

        score = float(sims[i])
        if score < min_sim:
            continue

        passage = index.passages[int(i)]

        # TF-IDF can return broad topical neighbors that are not actually
        # about the user's subject. This keeps cite-or-refuse strict.
        if index.embedder_name == "tfidf" and not _has_enough_lexical_overlap(
            question,
            passage,
        ):
            continue

        hits.append(
            {
                "passage": passage,
                "score": round(score, 4),
                "citation": _format_citation(passage),
                "rank": rank + 1,
            }
        )

    return hits


def compose_answer(
    question: str,
    retrieved: Sequence[dict[str, Any]],
    min_sim: float = 0.2,
) -> dict[str, Any]:
    """Compose the final answer dict for the UI."""
    answer = llm_compose(question, retrieved, min_sim=min_sim)
    refused = not retrieved or retrieved[0].get("score", 0.0) < min_sim

    citations: list[str] = []
    if not refused:
        citations = [hit.get("citation", "") for hit in retrieved]

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "passages": list(retrieved),
        "refused": refused,
        "disclaimer": DISCLAIMER,
    }


_DEFAULT_INDEX: Index | None = None
_DEFAULT_EMBEDDER: EmbeddingModel | None = None


def _get_default_pipeline(
    prefer: str = "sentence-transformers",
) -> tuple[Index, EmbeddingModel]:
    """Lazily build a default index from the bundled mini-corpus."""
    global _DEFAULT_INDEX, _DEFAULT_EMBEDDER

    if _DEFAULT_INDEX is not None and _DEFAULT_EMBEDDER is not None:
        return _DEFAULT_INDEX, _DEFAULT_EMBEDDER

    from .embedder import make_embedder

    try:
        from data.loader import load_mini_corpus
    except ImportError:
        from pathlib import Path
        import json

        corpus_path = (
            Path(__file__).resolve().parent.parent / "data" / "mini_corpus.json"
        )
        with corpus_path.open("r", encoding="utf-8") as fh:
            passages = json.load(fh)
    else:
        passages = load_mini_corpus()

    corpus_texts = [_passage_text(p) for p in passages]
    embedder = make_embedder(prefer=prefer, corpus_texts=corpus_texts)
    index = build_index(passages, embedder)

    _DEFAULT_INDEX = index
    _DEFAULT_EMBEDDER = embedder
    return index, embedder


def ask(
    question: str,
    top_k: int = 3,
    min_sim: float = 0.2,
    prefer: str = "sentence-transformers",
) -> dict[str, Any]:
    """One-call entry point used by the Streamlit app and notebook."""
    index, embedder = _get_default_pipeline(prefer=prefer)
    hits = retrieve(question, index, embedder, top_k=top_k, min_sim=min_sim)
    return compose_answer(question, hits, min_sim=min_sim)


def reset_default_pipeline() -> None:
    """Test helper: force the lazy default pipeline to rebuild next call."""
    global _DEFAULT_INDEX, _DEFAULT_EMBEDDER
    _DEFAULT_INDEX = None
    _DEFAULT_EMBEDDER = None