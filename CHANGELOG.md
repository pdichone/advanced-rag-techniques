# Changelog — Advanced RAG Techniques

Course-level changes, newest first. Dates are spoken in lectures, never rendered on slides.

## 2026-09-15

**Code**
- Chat model default changed from `gpt-3.5-turbo` to `gpt-5.6-luna` in `expansion_queries.py`, `expansion_answer.py`, `reranking.py`. OpenAI retires `gpt-3.5-turbo-0125` on 2026-10-23. Embeddings are local sentence-transformers and are unaffected.
- README rewritten to describe the course, the script-to-section map, setup and troubleshooting.
- `scikit-learn` added to `requirements.txt`; `dpr_technique.py` imports it and it was missing.

**Course (Udemy)**
- Section 3: new coding exercise "Classify Four Failure Traces", after "Deep Dive into Each Naive RAG Drawback".
- Section 3: new Role Play "Tell the PM Which Layer Broke", after that exercise.
- Section 8: new coding exercise "Implement a Bounded Retrieval Loop", after "Agentic RAG with LangGraph".
- Section 8: new Role Play "The CTO Read a Blog Post About GraphRAG", after "How to Choose an Advanced RAG Technique".

Day-30 read for the four new items: 2026-10-15 (participation and active minutes, matched exposure).
