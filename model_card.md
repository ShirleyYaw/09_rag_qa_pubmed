# Model Card — RAG PubMed Q&A Bot

**Project:** SciEncephalon AI Summer Intern 2026 · Project 09
**Maintainer:** Intern + mentor pair (rotating annually)
**Last updated:** at index-build time — regenerate when you swap in a new corpus or embedder

> :warning: **This is not medical advice.** The bot is an educational
> teaching artifact for high-school interns. It must never be used to
> make real clinical decisions for real patients.

---

## Intended use

- Educational demo of a Retrieval-Augmented Generation (RAG) pipeline.
- Showcase the **cite-or-refuse** design pattern: no good citation, no
  answer.
- Vehicle for teaching embeddings, cosine similarity, top-k retrieval,
  and modular ML pipeline design in healthcare AI.

## Out-of-scope use

- Real clinical Q&A or triage.
- Diagnosis, dosing, prescription, or insurance decisions.
- Any open-domain question answering (the corpus is intentionally
  closed and small).
- Crisis intervention. Mental-health questions surface the 988
  Lifeline; the bot is not a therapist.

## Inputs

- A single free-text question in English (1–2 sentences).

## Outputs

A dict with the following fields:

| Field | Type | Description |
|---|---|---|
| `question` | str | The original question, echoed back. |
| `answer` | str | The user-facing string. **Always contains "not medical advice".** Either contains at least one citation **or** the refusal phrase "I don't have a vetted source". |
| `citations` | list[str] | Zero or more pretty citations (`[source, ref id] url`). Empty when `refused=True`. |
| `passages` | list[dict] | Raw retrieval hits (passage, score, citation, rank). Useful for the UI. |
| `refused` | bool | True if retrieval was empty or below `min_sim`. The bot declined to answer rather than guess. |
| `disclaimer` | str | The full disclaimer text. |

## Training / corpus data

- **Default corpus:** `data/mini_corpus.json` — 20 vetted passages
  hand-curated from CDC, Mayo Clinic, WHO, NIH, NHLBI, NIMH, NIDDK, and
  AHA. Each passage has an `id`, `title`, `text`, `source`, and `url`.
- **Stretch goal:** real PubMed via `data/loader.py::load_pubmed`
  (currently a stub that raises `NotImplementedError`).

The mini-corpus is intentionally tiny so the whole pipeline can run
offline in a few hundred milliseconds. It is **not** clinically
representative; it exists only to demonstrate the architecture.

## Model architecture

1. **Embedder:** swappable.
   - Default: `sentence-transformers/all-MiniLM-L6-v2` (384-dim dense
     embeddings, cosine similarity).
   - Fallback: scikit-learn `TfidfVectorizer(ngram_range=(1,2))`
     densified and L2-normalized. Same interface. Used by all tests so
     the suite is offline and fast.
2. **Vector store:** in-memory numpy matrix. Cosine similarity is a
   plain matrix-vector dot product because vectors are L2-normalized.
3. **Retriever:** top-k by similarity, with a `min_sim` floor. Anything
   below the floor is dropped — empty result means "refuse".
4. **Generator:** templated stub (`llm_compose`). The stub does not
   call any external LLM. It assembles a grounded answer by quoting the
   retrieved passages with citation markers, or returns the refusal
   string. Wiring in a real LLM is a documented stretch goal — see
   `MENTOR_NOTES.md`.

## Evaluation

The notebook reports:

- Top-k retrieval scores per query (PyEcharts bar chart).
- Document × document similarity heatmap (PyEcharts heatmap).
- Hand-picked qualitative pass/fail on a small set of questions.

The notebook does **not** ship a quantitative cite-or-refuse evaluation
out of the box — that's the Week 4 milestone. Interns build a 20-answer
+ 10-refusal eval set and measure four numbers: recall, refusal rate,
false-answer rate (must be zero), and missed-answer rate.

### Quantitative Evaluation

The system already shows strong performance, with high usefulness, zero missed answers, and noticeably better retrieval consistency compared to TF‑IDF. The main limitation that remains is the occasional false answer, especially for questions that touch on a topic present in the corpus but don’t align precisely with the retrieved passage. For example, a question like “Can you permanently get rid of HIV?” may trigger retrieval on HIV and produce an answer even though the question requires a refusal. Addressing this will require tightening refusal logic, improving retrieval precision, and adding more explicit safety checks during generation. I attempted adjustments in pipeline.py, but the issue persists, suggesting that the fix needs to be applied deeper in the retrieval thresholding or safety‑filtering flow rather than only in the answer‑composition step.


## Known limitations

1. **Tiny mini-corpus (20 passages).** A real Q&A bot needs orders of
   magnitude more passages. The mini-corpus exists only so the notebook
   can run offline. Real performance numbers require a real corpus.
2. **English only.** Both backends assume English. Code-switched or
   non-English inputs will return weak, low-confidence retrievals — and
   then refuse.
3. **No multi-document synthesis.** The bot quotes each retrieved
   passage as its own bullet; it does not blend evidence across
   sources. Two passages that disagree will appear side-by-side without
   reconciliation. That's a documented stretch goal.
4. **No span-level grounding.** Citations point at whole passages, not
   the specific sentence. A retrieved passage might be on-topic but the
   exact sentence used to answer might be implicit. Span-level
   grounding is a stretch goal.
5. **`min_sim` is a single global threshold.** Real-world systems use
   per-query thresholds or learned calibration. The intern should
   notice this in Week 4 and write a sentence about it.
6. **TF-IDF fallback is lexical, not semantic.** It will miss
   paraphrases ("how can I avoid getting sick" might not match a
   passage titled "Hand washing"). The MiniLM backend handles synonyms
   better but requires a model download on first use.

## Safety behavior — cite-or-refuse

The "cite-or-refuse" rule is enforced in two places:

1. **`retrieve()`** filters out passages whose similarity is below
   `min_sim` and returns an empty list when nothing qualifies.
2. **`llm_compose()`** returns the refusal template whenever its input
   is empty or its top score is below `min_sim`.

Both checks are intentional — defense in depth. Removing either is a
correctness bug, not a refactor. Tests assert that every composed answer
either contains a citation **or** the refusal phrase "I don't have a
vetted source".

If you wire in a real LLM as a stretch goal, you must add a third
guardrail: post-generation verification that every cited id appears in
the retrieved passages. Otherwise the LLM can invent citations.

## Ethics

- Every user-facing string contains "not medical advice".
- The disclaimer references **988** (US suicide & crisis lifeline) and
  **911** (US emergency services).
- The bot **never invents a citation.** It only echoes the retrieved
  passages.
- The bot **refuses** when retrieval is weak. Refusal is a feature, not
  a bug.
- The Streamlit app does not send user queries to a third party.
- The LLM layer is a stub — when an intern wires in a real LLM
  (OpenAI, Anthropic, etc.) they must add a privacy note explaining
  what is sent to the provider.

## How to regenerate this model card

Re-run the notebook with your new corpus or embedder, run the Week 4
eval, then update this file by hand. Fill in the cite-or-refuse
numbers, the size and provenance of your corpus, and any new failure
modes you discovered. Industry-standard model-card auto-generation
(e.g. `model-card-toolkit`) is a stretch goal — see Project 08 for the
auto-generated reference shape.
