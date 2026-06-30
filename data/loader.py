"""
Corpus loaders for the RAG PubMed Q&A project.

The notebook follows the project's "real-or-synthetic fallback" rule: try the
real corpus first; if it isn't available, fall back to a small, curated
mini-corpus with the *same schema* so the rest of the pipeline runs unchanged.

This module exposes:

- ``load_mini_corpus()``: returns a list of dicts (one per passage) from the
  bundled ``mini_corpus.json`` file. Safe to call offline, no network.
- ``load_pubmed(query)``: live PubMed E-utilities query, with two heuristics
  layered on top to improve evidence quality. Neither of these is medical
  fact-checking — they are documented as heuristics, not guarantees:

    1. Evidence-level preference: results are tagged with a rough evidence
       tier (systematic_review / meta_analysis / clinical_trial / other) and
       sorted so higher tiers surface first. A systematic review still being
       wrong is possible; this just reflects the standard evidence hierarchy.

    2. Cross-corpus consistency flag: each PubMed passage is checked for
       lexical topic overlap against the vetted mini-corpus (CDC/Mayo/WHO/
       NIH/AHA). If a PubMed passage touches the same topic as a vetted
       passage, we record ``consistency: "topic_match"`` so the UI can show
       it alongside the vetted source rather than as a standalone claim. We
       deliberately do NOT attempt automated agree/disagree classification
       on free text — that's a job for a human reviewer or a much stronger
       model with explicit citation checking, and a false "verified" badge
       would be worse than no badge at all.

Each passage is a dict with the documented schema::

    {
        "id":     str,   # unique passage id, e.g. "D01"
        "title":  str,   # short headline
        "text":   str,   # the passage body (1-3 sentences)
        "source": str,   # the publishing organization, e.g. "CDC"
        "url":    str,   # canonical url for the citation
        "topic":  str,   # rough subject area, e.g. "cardiovascular"
    }

PubMed passages add two extra (optional) fields: ``evidence_level`` and
``consistency``. Code that only knows the original schema can ignore them.
"""

from __future__ import annotations

import json
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
import os

DEBUG = os.environ.get("PUBMED_LOADER_DEBUG", "") == "1"

_MINI_CORPUS_PATH = Path(__file__).resolve().parent / "mini_corpus.json"
_CACHE_PATH = Path(__file__).resolve().parent / "pubmed_cache.json"

REQUIRED_KEYS = {"id", "title", "text", "source", "url"}

_EVIDENCE_RANK = {
    "systematic_review": 0,
    "meta_analysis": 0,
    "clinical_trial": 1,
    "other": 2,
}

_PREFERRED_PUBTYPES = (
    '"systematic review"[pt] OR "meta-analysis"[pt] OR "randomized controlled trial"[pt]'
)

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "from", "that", "this", "it",
    "as", "at", "study", "studies", "patients", "results", "background",
    "what", "when", "where", "why", "how", "many", "much", "should", "can",
    "does", "do", "did", "are", "is", "could", "would", "will",
}

# Words that are *not* useful for a PubMed query even though they may be
# content words — they're too generic for medical search.
_QUERY_STOPWORDS = _STOPWORDS | {
    "people", "person", "thing", "things", "good", "bad", "best",
    "kind", "sort", "type", "types",
}


def load_mini_corpus(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the bundled vetted mini-corpus.

    Returns a list of dicts following the schema documented at the top of this
    module.
    """
    p = Path(path) if path is not None else _MINI_CORPUS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Mini-corpus file not found at {p}. The corpus ships with the "
            "project — re-clone the folder if it has disappeared."
        )

    with p.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list) or not records:
        raise ValueError(f"Corpus at {p} is empty or not a JSON list.")

    for i, rec in enumerate(records):
        missing = REQUIRED_KEYS - set(rec.keys())
        if missing:
            raise ValueError(
                f"Corpus record {i} (id={rec.get('id')!r}) missing keys: {missing}"
            )

    return records


def _load_cache() -> dict[str, list[dict[str, Any]]]:
    if _CACHE_PATH.exists():
        with _CACHE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_cache(cache: dict[str, list[dict[str, Any]]]) -> None:
    with _CACHE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _topic_overlap(passage_text: str, vetted_corpus: list[dict[str, Any]]) -> str | None:
    """Return the id of a vetted passage that lexically overlaps this text, if any."""
    passage_tokens = _tokenize(passage_text)
    if not passage_tokens:
        return None

    best_id, best_overlap = None, 0
    for vetted in vetted_corpus:
        vetted_tokens = _tokenize(f"{vetted['title']} {vetted['text']}")
        overlap = len(passage_tokens & vetted_tokens)
        if overlap > best_overlap:
            best_overlap, best_id = overlap, vetted["id"]

    return best_id if best_overlap >= 4 else None


def _classify_evidence_level(pubtype_elements: list[ET.Element]) -> str:
    types = {el.text for el in pubtype_elements if el.text}
    if "Systematic Review" in types:
        return "systematic_review"
    if "Meta-Analysis" in types:
        return "meta_analysis"
    if "Randomized Controlled Trial" in types or "Clinical Trial" in types:
        return "clinical_trial"
    return "other"


def load_pubmed(query: str, max_results: int = 20) -> list[dict[str, Any]]:
    """Load fresh PubMed abstracts via NCBI E-utilities, with caching."""
    cache = _load_cache()
    if query in cache:
        return cache[query]

    try:
        vetted_corpus = load_mini_corpus()
    except Exception:
        vetted_corpus = []

    try:
        boosted_query = f"({query}) AND ({_PREFERRED_PUBTYPES})"
        pmids = _esearch(boosted_query, max_results)
        if not pmids:
            pmids = _esearch(query, max_results)
        if not pmids:
            return []

        results = _efetch_and_annotate(pmids, vetted_corpus)

        if not results:
            return []

        results.sort(key=lambda r: _EVIDENCE_RANK.get(r["evidence_level"], 2))

        cache[query] = results
        _save_cache(cache)
        return results
    except Exception as exc:
        if DEBUG:
            print(f"[load_pubmed] falling back to mini-corpus, query={query!r}", file=sys.stderr)
            traceback.print_exc()
        return []


def _esearch(query: str, max_results: int) -> list[str]:
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "email": "you@school.edu",
    }
    response = requests.get(esearch_url, params=esearch_params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _efetch_and_annotate(
    pmids: list[str], vetted_corpus: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": "you@school.edu",
    }
    response = requests.get(efetch_url, params=efetch_params, timeout=15)
    response.raise_for_status()
    root = ET.fromstring(response.text)

    results: list[dict[str, Any]] = []
    for article in root.iter("PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None:
            continue
        pmid = pmid_elem.text

        art = article.find(".//Article")
        if art is None:
            continue

        title_el = art.find(".//ArticleTitle")
        title = (
            title_el.text.strip()
            if title_el is not None and title_el.text
            else f"PubMed Article {pmid}"
        )

        abstract_parts: list[str] = []
        for abstract_text in art.findall(".//AbstractText"):
            text = (abstract_text.text or "").strip()
            if text:
                abstract_parts.append(text)
        text = " ".join(abstract_parts).strip()
        if not text:
            continue

        pubtype_elements = article.findall(".//PublicationTypeList/PublicationType")
        evidence_level = _classify_evidence_level(pubtype_elements)
        consistency_match = _topic_overlap(f"{title}. {text}", vetted_corpus)

        passage = {
            "id": f"PMID{pmid}",
            "title": title,
            "text": text,
            "source": "PubMed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "topic": "pubmed",
            "evidence_level": evidence_level,
        }
        if consistency_match:
            passage["consistency"] = {
                "status": "topic_match",
                "vetted_id": consistency_match,
                "note": (
                    "Shares significant terms with a vetted passage — not a "
                    "claim that the two agree. Review both before citing."
                ),
            }
        results.append(passage)

    return results


# ----------------------------------------------------------------------
# New utilities
# ----------------------------------------------------------------------

def clear_pubmed_cache() -> int:
    """Delete the on-disk PubMed cache. Returns the number of queries dropped."""
    if not _CACHE_PATH.exists():
        return 0
    try:
        cache = _load_cache()
        n = len(cache)
    except Exception:
        n = 0
    try:
        _CACHE_PATH.unlink()
    except Exception:
        pass
    return n


def pubmed_cache_status() -> dict[str, Any]:
    """Return a summary of the on-disk PubMed cache for UI display."""
    if not _CACHE_PATH.exists():
        return {"exists": False, "queries": 0, "total_passages": 0, "queries_list": []}
    try:
        cache = _load_cache()
    except Exception:
        return {"exists": False, "queries": 0, "total_passages": 0, "queries_list": []}
    return {
        "exists": True,
        "queries": len(cache),
        "total_passages": sum(len(v) for v in cache.values()),
        "queries_list": sorted(cache.keys()),
    }


def corpus_coverage(
    question: str, vetted_corpus: list[dict[str, Any]], threshold: int = 3
) -> dict[str, Any]:
    """Cheap lexical check: does the vetted corpus already cover this question?"""
    q_tokens = _tokenize(question)
    if not q_tokens:
        return {"covered": False, "best_vetted_id": None,
                "overlap_count": 0, "query_terms": []}

    best_id, best_overlap = None, 0
    for v in vetted_corpus:
        v_tokens = _tokenize(f"{v['title']} {v['text']}")
        ov = len(q_tokens & v_tokens)
        if ov > best_overlap:
            best_overlap, best_id = ov, v["id"]

    return {
        "covered": best_overlap >= threshold,
        "best_vetted_id": best_id,
        "overlap_count": best_overlap,
        "query_terms": sorted(q_tokens),
    }


def build_pubmed_query(question: str) -> str:
    """Turn a free-text question into a compact PubMed search string.

    Strips stopwords, keeps the most informative terms, caps at 6 words.
    Kept here (next to the loader) so the UI and any CLI share one
    implementation.
    """
    words = [
        w for w in question.lower().split()
        if len(w) > 2 and w not in _QUERY_STOPWORDS
    ]
    if not words:
        return "health"
    return " ".join(words[:6])