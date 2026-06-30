# 09 — RAG PubMed Q&A

**SciEncephalon AI · Summer Intern 2026**
**You (the intern):** rising senior, June 1 → July 10, 2026.
**Coding deadline:** Friday, July 3, 2026. Final week (July 6 – July 10) is presentations.

> :warning: **This is not medical advice.** Everything in this project is
> an educational demo. The bot answers only from a small set of vetted
> public-health snippets and always cites them. If you're worried about
> your health, talk to your provider. For US emergencies call **911**.
> For mental-health crises call or text **988**.

---

## Goal

Build a **Retrieval-Augmented Generation (RAG)** Q&A bot that:

1. Answers consumer-health questions **only from a vetted corpus**
   (CDC, Mayo Clinic, WHO, NIH, AHA).
2. **Always cites** the document it drew from — by source name, document
   id, and URL.
3. **Refuses to answer** when nothing in the corpus is a good enough
   match. No citation → no answer. This is the "cite-or-refuse" rule.

The interesting part of this project is **not** the embedding model. It's
the guardrail: when the retriever fails, the generator must *refuse*. A
real LLM, given the same weak retrieval, will happily make something up.
Your job is to make sure this one never does.

The pipeline is intentionally modular:

```
question -> embed -> top-k retrieve -> compose answer w/ citations
                                    \-> or polite refusal
```

Each stage is swappable. You can replace the embedder, the retriever, or
the generator one at a time without breaking the others.

You'll come out of this project with three durable skills:

- How embeddings and retrieval actually work (cosine similarity, top-k,
  the difference between lexical and semantic matching).
- Why citation and refusal are first-class outputs, not afterthoughts.
- How to design a modular ML pipeline where you can swap a backend
  without touching downstream code.

---

## What's in this folder

```
09_rag_qa_pubmed/
  09_rag_qa_pubmed.ipynb       # the notebook you start with
  README.md                     # you are here
  MENTOR_NOTES.md               # for your mentor — feel free to read
  model_card.md                 # the model's "nutrition label"
  requirements.txt              # pip dependencies
  src/embedder.py               # sentence-transformers + TF-IDF fallback
  src/llm_stub.py               # templated grounded answer / polite refusal
  src/pipeline.py               # build_index, retrieve, compose_answer, ask
  app/streamlit_app.py          # the working UI demo
  tests/test_pipeline.py        # tests (run them often!)
  data/loader.py                # mini-corpus loader + PubMed stretch stub
  data/mini_corpus.json         # 20 vetted passages from CDC/Mayo/WHO/NIH/AHA
```

---

## Quick start

From the repo root (one directory up from this README):

```bash
# 1. Install deps (use a virtualenv or conda env)
pip install -r 09_rag_qa_pubmed/requirements.txt

# 2. Open the notebook and Run All
jupyter notebook 09_rag_qa_pubmed/09_rag_qa_pubmed.ipynb

# 3. Run the tests (they must pass before you commit)
python -m pytest 09_rag_qa_pubmed/tests/ -x --tb=short

# 4. Launch the Streamlit demo
streamlit run 09_rag_qa_pubmed/app/streamlit_app.py
```

The notebook and tests are designed to run **offline** on the bundled
mini-corpus using the **TF-IDF fallback**. You only need a network on
your first sentence-transformers call (to download the MiniLM model).
Tests never download anything.

---

## Troubleshooting & Tips
### Not getting results from PubMed?
If you ask a question outside the core vetted corpus and get a refusal (or no sources), your min_sim threshold in the sidebar is likely too high. Try lowering it (e.g., to 0.05) to allow slightly weaker matches from PubMed abstracts to pass the filter.
### TF-IDF Stopword Penalty:
Filler words like "what", "the", or "is" in user queries can artificially drag down cosine similarity scores to zero. The Streamlit app handles this automatically by stripping stopwords before passing the query to the retriever.
### Black screen on launch?
Streamlit hides Python syntax errors. If the app is blank, run python app/streamlit_app.py directly in your terminal to see the actual traceback, then clear your __pycache__ folders.