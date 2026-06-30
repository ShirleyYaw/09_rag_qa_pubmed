"""
Streamlit demo for the RAG PubMed Q&A bot.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

# Import base functions
from data.loader import load_mini_corpus, load_pubmed  # noqa: E402
from src.embedder import make_embedder  # noqa: E402
from src.pipeline import build_index, retrieve  # noqa: E402

# Defensive import to handle Python caching issues or missing functions
try:
    from src.llm_stub import DISCLAIMER, llm_compose, get_model_status  # noqa: E402
except ImportError:
    from src.llm_stub import DISCLAIMER, llm_compose  # noqa: E402
    def get_model_status():  # type: ignore
        return {"loaded": False, "name": None, "candidates": []}


# --------------------------------------------------------------------- #
# Helper utilities 
# --------------------------------------------------------------------- #

_CACHE_PATH = PROJECT_ROOT / "data" / "pubmed_cache.json"

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "from", "that", "this", "it",
    "as", "at", "study", "studies", "patients", "results", "background",
    "what", "when", "where", "why", "how", "many", "much", "should", "can",
    "does", "do", "did", "are", "is", "could", "would", "will",
}
_QUERY_STOPWORDS = _STOPWORDS | {
    "people", "person", "thing", "things", "good", "bad", "best",
    "kind", "sort", "type", "types",
}

def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}

def _clean_query_for_tfidf(question: str) -> str:
    words = question.lower().split()
    cleaned = [w for w in words if len(w) > 2 and w not in _STOPWORDS]
    return " ".join(cleaned) if cleaned else question

def build_pubmed_query(question: str) -> str:
    words = [w for w in question.lower().split() if len(w) > 2 and w not in _QUERY_STOPWORDS]
    return " ".join(words[:6]) if words else "health"

def clear_pubmed_cache() -> int:
    if not _CACHE_PATH.exists(): return 0
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as fh: cache = json.load(fh)
        n = len(cache)
    except: n = 0
    try: _CACHE_PATH.unlink()
    except: pass
    return n

def pubmed_cache_status() -> dict:
    if not _CACHE_PATH.exists(): return {"exists": False, "queries": 0, "total_passages": 0, "queries_list": []}
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as fh: cache = json.load(fh)
    except: return {"exists": False, "queries": 0, "total_passages": 0, "queries_list": []}
    return {"exists": True, "queries": len(cache), "total_passages": sum(len(v) for v in cache.values()), "queries_list": sorted(cache.keys())}

def corpus_coverage(question: str, vetted_corpus: list, threshold: int = 3) -> dict:
    q_tokens = _tokenize(question)
    if not q_tokens: return {"covered": False, "best_vetted_id": None, "overlap_count": 0, "query_terms": []}
    best_id, best_overlap = None, 0
    for v in vetted_corpus:
        v_tokens = _tokenize(f"{v['title']} {v['text']}")
        ov = len(q_tokens & v_tokens)
        if ov > best_overlap: best_overlap, best_id = ov, v["id"]
    return {"covered": best_overlap >= threshold, "best_vetted_id": best_id, "overlap_count": best_overlap, "query_terms": sorted(q_tokens)}

def _evidence_badge(p: dict) -> str:
    title = p.get("title", "").lower()
    text = p.get("text", "").lower()
    if "guideline" in title or "guideline" in text: return "Guideline"
    if "systematic review" in title or "meta-analysis" in title: return "Systematic Review"
    if "clinical trial" in title or "randomized" in text: return "Clinical Trial"
    if "cohort" in title or "study" in title: return "Observational Study"
    return "General Info"

def _clean_answer_text(text: str) -> str:
    if not text: return ""
    text = text.strip()
    if text.lower().startswith("answer:"): text = text[len("answer:"):].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

_JARGON_MAP = {
    "hypertension": "high blood pressure", "cardiovascular": "heart and blood vessel",
    "myocardial infarction": "heart attack", "diabetes mellitus": "diabetes",
    "pulmonary": "lung", "onset": "start", "contraindicated": "not safe to use",
    "contraindications": "reasons it is not safe", "efficacy": "how well it works",
    "effectiveness": "how well it works in real life", "etiology": "cause",
    "exacerbate": "make worse", "exacerbation": "worsening", "adherence": "following the plan",
    "utilize": "use", "utilization": "use", "subsequently": "later", "physician": "doctor",
    "physicians": "doctors", "manifestations": "signs", "indications": "reasons to use",
    "administered": "given", "administer": "give", "beneficial": "helpful", "sufficient": "enough",
    "demonstrate": "show", "approximately": "about", "particularly": "especially",
    "primarily": "mostly", "individuals": "people", "additionally": "also", "furthermore": "also",
    "consequently": "as a result", "therefore": "so", "intervention": "treatment or action",
    "interventions": "treatments or actions", "outcomes": "results", "prospective": "forward-looking",
    "retrospective": "looking at past records", "statistically significant": "strong enough to not be just by chance",
    "prevalence": "how common something is", "incidence": "number of new cases", "mortality": "death rate",
    "morbidity": "illness rate", "comorbidity": "having other illnesses at the same time",
    "comorbidities": "other illnesses at the same time", "prognosis": "likely outcome",
    "symptomatic": "showing signs of illness", "asymptomatic": "showing no signs of illness",
    "pathogenesis": "how a disease develops", "therapeutic": "treatment-related",
    "pharmacological": "drug-related", "baseline": "starting point", "cohort": "group",
    "methodology": "methods", "demographic": "population traits", "demographics": "population traits like age or gender",
    "variables": "factors", "correlation": "relationship", "hypothesis": "proposed idea",
    "randomized": "randomly picked", "placebo": "fake treatment (like a sugar pill)",
    "double-blind": "where neither the patients nor the doctors know who got the real treatment",
    "clinical trial": "research study testing a treatment", "systematic review": "study that looks at all the research on a topic",
    "meta-analysis": "study that combines data from many other studies", "abstain": "avoid completely",
    "abstinence": "avoiding completely", "sedentary": "sitting around, not moving much",
    "mitigate": "reduce or lessen", "mitigation": "reducing or lessening", "impairment": "damage or loss of function",
    "deficit": "lack or shortage", "regimen": "plan or routine", "adjunctive": "added to help the main treatment",
    "alleviate": "relieve or reduce pain/symptoms", "deterioration": "getting worse",
    "prophylactic": "used to prevent disease", "pathophysiology": "how a disease changes normal body functions",
    "multivariate": "looking at many factors at once", "longitudinal": "following people over a long time",
    "cross-sectional": "looking at a single point in time", "bioavailability": "how much of a drug gets into the bloodstream",
    "adverse events": "bad side effects", "paramount": "most important", "elucidate": "explain or make clear",
    "underscore": "highlight or show importance", "necessitate": "make necessary", "augment": "add to or increase",
    "diminish": "decrease or reduce", "facilitate": "make easier", "pertaining to": "related to",
    "in lieu of": "instead of", "with regard to": "regarding", "aforementioned": "mentioned earlier",
}

def _adapt_text_for_grade(text: str, grade: int) -> str:
    if not text: return text
    for complex_term, simple_term in _JARGON_MAP.items():
        text = re.sub(rf'\b{re.escape(complex_term)}\b', simple_term, text, flags=re.IGNORECASE)
    if grade >= 9: return text

    max_words = 12 if grade <= 5 else 20
    sentences = re.split(r'(?<=[.!?])\s+', text)
    final_sentences = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            final_sentences.append(sentence)
            continue

        if grade <= 5:
            text_mod = re.sub(r',\s+(which|who|where|while|although|though)\s+', r'. \1 ', sentence)
            text_mod = re.sub(r',\s+(and|but|or|so)\s+', r'. \1 ', text_mod)
            text_mod = re.sub(r';\s*', '. ', text_mod)
            parts = re.split(r'(?<=[.!?])\s+', text_mod)
            for part in parts:
                part = part.strip()
                if not part: continue
                part = part[0].upper() + part[1:]
                if not part.endswith('.'): part += '.'
                final_sentences.append(part)
        else:
            split_patterns = [r'\s+and\s+', r'\s+but\s+', r'\s+or\s+', r'\s+which\s+', r'\s+so\s+', r';\s*']
            split_done = False
            for pattern in split_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    parts = re.split(pattern, sentence, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        part1 = parts[0].strip()
                        part2 = parts[1].strip()
                        if part2 and part2[0].islower(): part2 = part2[0].upper() + part2[1:]
                        if not part1.endswith('.'): part1 += '.'
                        final_sentences.append(part1)
                        final_sentences.append(_adapt_text_for_grade(part2, grade).strip())
                        split_done = True
                        break
            if not split_done: final_sentences.append(sentence)
    return " ".join(final_sentences)


# --------------------------------------------------------------------- #
# Cached pipeline pieces
# --------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Building index...")
def _build_pipeline(prefer: str):
    passages = load_mini_corpus()
    corpus_texts = [f"{p['title']}. {p['text']}" for p in passages]
    embedder = make_embedder(prefer=prefer, corpus_texts=corpus_texts)
    index = build_index(passages, embedder)
    return passages, index, embedder

def _retrieve_with_pubmed_fallback(question: str, passages: list[dict], embedder_name: str, top_k: int, min_sim: float, force_pubmed: bool):
    coverage = corpus_coverage(question, passages)
    fetched_pubmed, pubmed_query, pubmed_count = False, "", 0
    used_passages = list(passages)
    need_pubmed = force_pubmed or not coverage["covered"]

    if need_pubmed:
        pubmed_query = build_pubmed_query(question)
        try: fresh = load_pubmed(pubmed_query, max_results=8)
        except Exception: fresh = []
        seen_ids = {p.get("id") for p in used_passages}
        for p in fresh:
            if p.get("id") not in seen_ids:
                used_passages.append(p)
                seen_ids.add(p.get("id"))
        pubmed_count = len(fresh)
        fetched_pubmed = True

    corpus_texts = [f"{p['title']}. {p['text']}" for p in used_passages]
    embedder = make_embedder(prefer=embedder_name, corpus_texts=corpus_texts)
    index = build_index(used_passages, embedder)
    search_query = _clean_query_for_tfidf(question)
    hits = retrieve(search_query, index, embedder, top_k=top_k, min_sim=min_sim)

    return {"hits": hits, "used_passages": used_passages, "pubmed_query": pubmed_query, "pubmed_count": pubmed_count, "coverage": coverage, "fetched_pubmed": fetched_pubmed}


# --------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------- #

def _render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("### ⚕️ RAG Q&A Control Panel")
        st.caption("**SciEncephalon AI · Intern 09**")
        st.divider()
        
        with st.expander("Model & Retrieval", expanded=True):
            backend = st.radio("Embedder", ["sentence-transformers", "tfidf"], index=1, label_visibility="collapsed")
            
            col1, col2 = st.columns(2)
            with col1: reading_level = st.slider("Grade Level", 3, 12, 8)
            with col2: top_k = st.slider("Top K", 1, 5, 3)
            
            min_sim = st.slider("Min Similarity", 0.0, 0.8, 0.2, step=0.05, help="Lower this (e.g. 0.05) if PubMed abstracts are being ignored.")

        with st.expander("Local LLM", expanded=False):
            status = get_model_status()
            if status["loaded"]:
                st.success(f"Loaded: `{status['name']}`")
            else:
                st.info("No local LLM. Using safe templates.")
            use_local_llm = st.checkbox("Use Local LLM", value=False, disabled=not status["loaded"])

        with st.expander("PubMed Settings", expanded=False):
            cache_info = pubmed_cache_status()
            st.metric("Cached Passages", cache_info["total_passages"])
            if st.button("Clear Cache", use_container_width=True):
                n = clear_pubmed_cache()
                st.success(f"Cleared {n} queries.")
                st.rerun()
            force_pubmed = st.checkbox("Force PubMed Fetch", value=False)

        st.divider()
        st.warning("⚠️ **Not medical advice.** Verify the information with a clinician.")

    return {"backend": backend, "reading_level": reading_level, "top_k": top_k, "min_sim": min_sim, "use_local_llm": use_local_llm, "force_pubmed": force_pubmed}


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="RAG PubMed Q&A", page_icon="🔎", layout="centered")

    cfg = _render_sidebar()
    passages, index, embedder = _build_pipeline(prefer=cfg["backend"])

    st.title("RAG PubMed Q&A Bot")
    st.caption(
        "Ask a consumer-health question. The bot answers only from vetted "
        "sources (CDC, Mayo, WHO, NIH, AHA). When the topic isn't in the "
        "corpus, the RAG pulls fresh PubMed abstracts. It always cites sources and "
        "politely refuses when it cannot provide a reliable answer. "
        "Remember: this is not medical advice."
    )

    with st.expander(f"Corpus info ({index.size} passages, embedder: {embedder.name})"):
        sources = sorted({p["source"] for p in passages})
        topics = sorted({p.get("topic", "?") for p in passages})
        st.write("**Sources:**", ", ".join(sources))
        st.write("**Topics:**", ", ".join(topics))

    default_question = "How many hours of sleep do teenagers need?"
    question = st.text_input(
        "Your question:",
        value=default_question,
        help="Ask a general health question — e.g. 'What is type 2 diabetes?'",
    )

    with st.expander("PubMed utilities", expanded=False):
        if question.strip():
            pre_cov = corpus_coverage(question, passages)
            st.write("**Pre-query corpus coverage check**")
            st.write(f"- Query terms extracted: {', '.join(pre_cov['query_terms']) or '(none)'}")
            st.write(f"- Shared significant terms with vetted corpus: {pre_cov['overlap_count']}")
            st.write(
                f"- Covered by vetted corpus: "
                f"{'yes' if pre_cov['covered'] else 'no — will likely fetch PubMed'}"
            )
        else:
            st.info("Type a question above to see a corpus coverage analysis.")

        pubmed_result_placeholder = st.empty()

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.error("Please type a question first.")
            return

        with st.status("Analyzing your question...", expanded=True) as status:
            st.write("Searching the corpus...")
            result = _retrieve_with_pubmed_fallback(
            question=question, passages=passages, embedder_name=embedder.name,
            top_k=cfg["top_k"], min_sim=cfg["min_sim"], force_pubmed=cfg["force_pubmed"],
            )
        hits = result["hits"]
        
        if result["fetched_pubmed"]:
            st.write(f"Fetched {result['pubmed_count']} abstracts from PubMed.")
            
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:8px; margin:8px 0 12px 0; font-size:16px; color:#ffffff;">
                    <strong>Generating grounded answer</strong>
                    <span style="display:inline-flex; align-items:center; gap:3px; color:#ffffff;">
                        <span style="animation: bob 0.9s infinite;">●</span>
                        <span style="animation: bob 0.9s infinite 0.2s;">●</span>
                        <span style="animation: bob 0.9s infinite 0.4s;">●</span>
                    </span>
                </div>
                <style>
                @keyframes bob {
                    0%, 80%, 100% { transform: translateY(0); opacity: 0.6; }
                    40% { transform: translateY(-5px); opacity: 1; }
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

        answer = llm_compose(
            question=question, retrieved=hits, min_sim=cfg["min_sim"], use_local_llm=cfg["use_local_llm"],
        )
        loading_placeholder.empty()
        status.update(label="Analysis Complete", state="complete", expanded=False)

        with pubmed_result_placeholder.container():
            if result["fetched_pubmed"]:
                st.markdown("**Live PubMed Fetch Results**")
                st.write(
                    f"- PubMed query sent: `{result['pubmed_query']}` "
                    f"({result['pubmed_count']} abstract(s) fetched)"
                )
                if result["pubmed_count"] > 0:
                    st.success("PubMed abstracts successfully merged into the retrieval pool.")
            else:
                st.markdown("**Live PubMed Fetch Results**")
                st.write("- PubMed query: _skipped (corpus already covers topic or not forced)_")

        st.markdown("#### Sources")
        if not hits:
            st.info(
                "Nothing in the corpus matched well enough to answer. "
                "Try a different phrasing, or enable 'Always fetch PubMed'."
            )
        else:
            for hit in hits:
                p = hit["passage"]
                score = hit["score"]
                origin = "Vetted corpus" if p["source"] != "PubMed" else "PubMed (fresh)"
                with st.container():
                    st.markdown(f"**{p['source']} · {p['title']}**")
                    badge = _evidence_badge(p)
                    tags = [f"`{origin}`"]
                    if badge:
                        tags.append(f"`{badge}`")
                    if p.get("consistency"):
                        tags.append("`topic-match to vetted source`")
                    st.caption(
                        f"Similarity: {score:.3f} · Ref: {p['id']} · "
                        f"{' · '.join(tags)}"
                    )
                    if p.get("url"):
                        st.caption(f"URL: {p['url']}")
                    st.markdown("---")

        st.markdown("#### Answer")
        cleaned = _clean_answer_text(answer)
        adapted = _adapt_text_for_grade(cleaned, cfg["reading_level"])

        # Fix answer spacing with HTML line breaks
        html_adapted = adapted.replace("\n", "<br>")

        if "I don't have a vetted source" in answer:
            st.warning(adapted)
        else:
            st.markdown(
                f"<div style='padding:12px 14px; border-left:4px solid #4CAF50; "
                f"background-color:rgba(76,175,80,0.10); border-radius:8px;'>"
                f"{html_adapted}</div>",
                unsafe_allow_html=True,
            )

        st.caption(
            f"Based on {len(hits)} source passage(s). "
            f"{'Local LLM used for paraphrasing.' if cfg['use_local_llm'] else 'Templated generator.'}"
        )

        st.markdown("---")
        st.markdown(f":warning: **Disclaimer.** {DISCLAIMER}")


if __name__ == "__main__":
    main()