"""
Tests for the RAG PubMed Q&A pipeline.

Run from the repo root:

    python -m pytest 09_rag_qa_pubmed/tests/ -x --tb=short

These tests must:
- Pass offline (no network, no HuggingFace download).
- Complete in well under 30 seconds.
- Use the TF-IDF embedder so we never depend on sentence-transformers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ and data/ importable when pytest is run from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.streamlit_app import _adapt_text_for_grade, _build_pubmed_query  # noqa: E402
from data.loader import load_mini_corpus  # noqa: E402
from src.embedder import TFIDFEmbedder, make_embedder  # noqa: E402
from src.llm_stub import DISCLAIMER, REFUSAL_TEMPLATE  # noqa: E402
from src.pipeline import (  # noqa: E402
    Index,
    build_index,
    compose_answer,
    reset_default_pipeline,
    retrieve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def passages() -> list[dict]:
    """The bundled vetted mini-corpus."""
    corpus = load_mini_corpus()
    assert len(corpus) >= 15, "Corpus should contain at least a small set of passages"
    return corpus


@pytest.fixture(scope="module")
def corpus_texts(passages: list[dict]) -> list[str]:
    return [f"{p['title']}. {p['text']}" for p in passages]


@pytest.fixture(scope="module")
def tfidf_embedder(corpus_texts: list[str]) -> TFIDFEmbedder:
    return TFIDFEmbedder().fit(corpus_texts)


@pytest.fixture(scope="module")
def index(passages: list[dict], tfidf_embedder: TFIDFEmbedder) -> Index:
    return build_index(passages, tfidf_embedder)


@pytest.fixture(autouse=True)
def _reset_default_pipeline_between_tests():
    reset_default_pipeline()
    yield
    reset_default_pipeline()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

def test_tfidf_embedder_returns_ndarray_shape(corpus_texts):
    """TFIDFEmbedder.embed(['hello', 'world']) returns ndarray of (2, dim)."""
    emb = TFIDFEmbedder().fit(corpus_texts)
    vecs = emb.embed(["hello", "world"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.ndim == 2
    assert vecs.shape[0] == 2
    assert vecs.shape[1] > 0
    # L2 normalization is part of the contract: every row should be unit norm
    # (or all-zero for empty/OOV strings — both fine).
    norms = np.linalg.norm(vecs, axis=1)
    for n in norms:
        assert np.isclose(n, 0.0) or np.isclose(n, 1.0, atol=1e-5)


def test_tfidf_embedder_must_be_fitted_before_embed():
    """Forgetting to fit() should raise a clear error."""
    emb = TFIDFEmbedder()
    with pytest.raises(RuntimeError, match="not fitted"):
        emb.embed(["hello"])


def test_make_embedder_falls_back_to_tfidf(corpus_texts):
    """If sentence-transformers isn't usable, we should still get something."""
    # Force the TF-IDF path explicitly via prefer="tfidf".
    emb = make_embedder(prefer="tfidf", corpus_texts=corpus_texts)
    assert emb.name == "tfidf"
    vecs = emb.embed(["sleep is good"])
    assert vecs.shape == (1, len(emb._vectorizer.get_feature_names_out()))


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def test_retrieve_returns_top_k(index, tfidf_embedder):
    """retrieve() returns up to top_k passages, each with a score."""
    hits = retrieve(
        "How can I prevent the spread of germs by washing my hands?",
        index,
        tfidf_embedder,
        top_k=3,
        min_sim=0.0,
    )
    assert 1 <= len(hits) <= 3
    # Top-1 should be the hand-washing passage.
    assert hits[0]["passage"]["id"] == "D01"
    assert hits[0]["score"] >= hits[-1]["score"]
    # Schema check
    for hit in hits:
        assert "passage" in hit
        assert "score" in hit
        assert "citation" in hit
        assert isinstance(hit["score"], float)


def test_retrieve_filters_by_min_sim(index, tfidf_embedder):
    """A nonsensical question should drop everything below min_sim."""
    hits = retrieve(
        "qwerty plumbus zxcvbn floooop",
        index,
        tfidf_embedder,
        top_k=3,
        min_sim=0.2,
    )
    # TF-IDF on out-of-vocab tokens gives a zero query vector, so all sims
    # will be ~0 and nothing should clear the floor.
    assert hits == []


def test_retrieve_empty_question_returns_empty(index, tfidf_embedder):
    """An empty/whitespace question is treated as no retrieval."""
    assert retrieve("", index, tfidf_embedder) == []
    assert retrieve("   ", index, tfidf_embedder) == []


# ---------------------------------------------------------------------------
# Compose: cite-or-refuse contract
# ---------------------------------------------------------------------------

def test_compose_answer_includes_citation_on_good_retrieval(index, tfidf_embedder):
    """A good question should produce an answer with at least one citation."""
    hits = retrieve(
        "What are the symptoms of a heart attack?",
        index,
        tfidf_embedder,
        top_k=3,
        min_sim=0.0,
    )
    result = compose_answer("What are the symptoms of a heart attack?", hits)
    assert result["refused"] is False
    assert len(result["citations"]) >= 1
    assert "not medical advice" in result["answer"].lower()
    # Citation marker must appear in the answer body.
    assert "ref [" in result["answer"]


def test_compose_answer_refuses_on_irrelevant_question(index, tfidf_embedder):
    """An irrelevant question must trigger a refusal, not a hallucinated answer."""
    hits = retrieve(
        "plumbus zxcvbn floooop qwerty",
        index,
        tfidf_embedder,
        top_k=3,
        min_sim=0.2,
    )
    result = compose_answer("plumbus zxcvbn floooop qwerty", hits)
    assert result["refused"] is True
    assert result["citations"] == []
    assert "I don't have a vetted source" in result["answer"]
    assert "not medical advice" in result["answer"].lower()


def test_compose_answer_refuses_on_empty_retrieval(index, tfidf_embedder):
    """An empty retrieval list always means refuse."""
    result = compose_answer("any question at all", [])
    assert result["refused"] is True
    assert result["citations"] == []
    assert "not medical advice" in result["answer"].lower()


def test_every_answer_has_citation_or_refusal(index, tfidf_embedder):
    """Cite-or-refuse: every composed answer must have one or the other."""
    questions = [
        "How much sleep do teenagers need?",
        "What is type 2 diabetes?",
        "asdfghjkl qwerty 12345",  # nonsense
        "How do I prevent skin cancer?",
        "blue moon banana telephone",  # nonsense
    ]
    for q in questions:
        hits = retrieve(q, index, tfidf_embedder, top_k=3, min_sim=0.2)
        result = compose_answer(q, hits, min_sim=0.2)
        has_citation = len(result["citations"]) >= 1
        is_refusal = result["refused"] is True
        assert has_citation or is_refusal, (
            f"Question {q!r} produced an answer with no citation AND no refusal — "
            "this violates the cite-or-refuse invariant."
        )
        assert "not medical advice" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Corpus sanity
# ---------------------------------------------------------------------------

def test_mini_corpus_has_required_fields(passages):
    """Every passage must have id/title/text/source/url."""
    for p in passages:
        for key in ("id", "title", "text", "source", "url"):
            assert key in p, f"passage missing {key}: {p}"
            assert p[key], f"passage has empty {key}: {p}"


def test_disclaimer_constants_are_safe():
    """The disclaimer + refusal templates must mention 988 and 911."""
    assert "not medical advice" in DISCLAIMER.lower()
    assert "988" in DISCLAIMER
    assert "911" in DISCLAIMER
    assert "vetted source" in REFUSAL_TEMPLATE


def test_adapt_text_for_grade_simplifies_for_lower_levels():
    """Lower grade levels should produce shorter, simpler wording."""
    text = "This explanation may be useful for understanding symptoms and treatment options."
    simple = _adapt_text_for_grade(text, 3)
    assert len(simple.split()) <= len(text.split())
    assert "This explanation" in simple or "This" in simple


def test_build_pubmed_query_removes_stop_words_and_keeps_keywords():
    """The PubMed query should preserve topic keywords while dropping filler words."""
    query = _build_pubmed_query("What are the symptoms of type 2 diabetes?")
    assert "symptoms" in query
    assert "diabetes" in query
    assert "what" not in query
    assert "the" not in query
