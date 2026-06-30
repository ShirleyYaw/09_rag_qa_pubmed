"""Public exports for the RAG PubMed Q&A project."""

from .embedder import (
    EmbeddingModel,
    SentenceTransformerEmbedder,
    TFIDFEmbedder,
    make_embedder,
)
from .llm_stub import DISCLAIMER, REFUSAL_TEMPLATE, llm_compose
from .pipeline import (
    Index,
    ask,
    build_index,
    compose_answer,
    reset_default_pipeline,
    retrieve,
    should_force_refuse,
)

__all__ = [
    "DISCLAIMER",
    "EmbeddingModel",
    "Index",
    "REFUSAL_TEMPLATE",
    "SentenceTransformerEmbedder",
    "TFIDFEmbedder",
    "ask",
    "build_index",
    "compose_answer",
    "llm_compose",
    "make_embedder",
    "reset_default_pipeline",
    "retrieve",
    "should_force_refuse",
]
