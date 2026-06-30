"""
Embedder layer for the RAG PubMed Q&A project.

The retriever doesn't care *how* we turn text into vectors — only that we can
do it consistently. This module ships two interchangeable backends behind a
single ``EmbeddingModel`` interface:

- ``SentenceTransformerEmbedder`` -- uses
  ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim dense embeddings).
- ``TFIDFEmbedder`` -- plain scikit-learn TF-IDF, used as a fallback whenever
  ``sentence-transformers`` is not installed or the network can't reach the
  HuggingFace mirror.

Use the factory ``make_embedder()`` to get the best backend that actually
works in the current environment. The notebook + tests + Streamlit app all
go through that factory so swapping backends is invisible to the rest of
the pipeline.

A note on the interface
-----------------------
Every embedder has::

    .fit(corpus_texts)              # required for TF-IDF; no-op for ST
    .embed(texts) -> np.ndarray     # shape (n_texts, dim), L2-normalized
    .name -> str                    # human-readable backend name

We L2-normalize embeddings so cosine similarity collapses to a plain dot
product downstream. That keeps the retriever simple.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; safe against zero rows (rare for TF-IDF)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class EmbeddingModel(ABC):
    """Abstract embedder interface.

    Subclasses must implement ``embed`` and expose a human-readable ``name``.

    ``fit`` is part of the contract because TF-IDF needs to see the corpus
    before it can transform anything. For embedders that don't need fitting
    (e.g. sentence-transformers), ``fit`` is a no-op.
    """

    name: str = "abstract"

    def fit(self, texts: Sequence[str]) -> "EmbeddingModel":
        """Optional. TF-IDF overrides this; dense embedders no-op."""
        return self

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Return an ``(n_texts, dim)`` L2-normalized float32 ndarray."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Backend 1 — sentence-transformers (preferred)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(EmbeddingModel):
    """Dense embeddings from ``sentence-transformers/all-MiniLM-L6-v2``.

    Loaded lazily so the import doesn't trigger a HuggingFace download when
    only the TF-IDF path is used (i.e. in tests).

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed.
    RuntimeError
        If the model can't be loaded (e.g. no network on first use and no
        cached copy).
    """

    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Either `pip install "
                "sentence-transformers` or use TFIDFEmbedder / make_embedder "
                "(which falls back automatically)."
            ) from exc

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001 — surface anything as a clear runtime error
            raise RuntimeError(
                f"Failed to load sentence-transformers model {model_name!r}. "
                "If you have no network and no cached copy, fall back to "
                "TFIDFEmbedder."
            ) from exc

        self.name = model_name

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,  # already L2-normalized
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Backend 2 — TF-IDF (always-available fallback)
# ---------------------------------------------------------------------------

class TFIDFEmbedder(EmbeddingModel):
    """Sparse TF-IDF embeddings, densified and L2-normalized.

    This is the offline-safe path. It doesn't require any downloads and is
    fast enough for the project's mini-corpus (~20 passages). It's also the
    backend used by the test suite to keep CI fast and hermetic.

    Notes
    -----
    - We densify the TF-IDF matrix because the rest of the pipeline expects
      ndarrays. For corpora of a few hundred passages this is harmless.
    - We must call ``fit(corpus_texts)`` before ``embed`` so the vocabulary
      is known. The ``make_embedder`` factory does this for you.
    """

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1):
        # Local import keeps ``import embedder`` cheap.
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: WPS433

        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, min_df=min_df, lowercase=True
        )
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TFIDFEmbedder":
        texts = list(texts)
        if not texts:
            raise ValueError("TFIDFEmbedder.fit needs at least one text.")
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not self._fitted:
            raise RuntimeError(
                "TFIDFEmbedder is not fitted. Call .fit(corpus_texts) first "
                "(or use make_embedder() which does this for you)."
            )
        if not texts:
            dim = len(self._vectorizer.get_feature_names_out())
            return np.zeros((0, dim), dtype=np.float32)
        sparse = self._vectorizer.transform(texts)
        dense = sparse.toarray().astype(np.float32, copy=False)
        return _l2_normalize(dense)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_embedder(
    prefer: str = "sentence-transformers",
    corpus_texts: Sequence[str] | None = None,
) -> EmbeddingModel:
    """Return the best embedder that actually works here.

    Parameters
    ----------
    prefer:
        Either ``"sentence-transformers"`` (default) or ``"tfidf"``. If the
        preferred backend can't be loaded (e.g. import fails, no network on
        first use), we fall back to TF-IDF and print a one-line note so the
        intern can see what happened.
    corpus_texts:
        If you pick TF-IDF (directly or via fallback) we need to fit the
        vocabulary. Pass the corpus passages here. For sentence-transformers
        this argument is ignored.

    Returns
    -------
    EmbeddingModel
        Ready to call ``.embed(texts)`` on.
    """
    prefer = prefer.lower().strip()

    if prefer in ("sentence-transformers", "st", "minilm"):
        try:
            return SentenceTransformerEmbedder()
        except (ImportError, RuntimeError) as exc:
            print(
                "[embedder] sentence-transformers unavailable "
                f"({type(exc).__name__}); falling back to TF-IDF."
            )

    # TF-IDF path (either explicitly requested or fallback)
    embedder = TFIDFEmbedder()
    if corpus_texts is None:
        raise ValueError(
            "TF-IDF needs a corpus to fit on. Pass corpus_texts= to "
            "make_embedder, or call .fit(corpus_texts) yourself."
        )
    embedder.fit(corpus_texts)
    return embedder
