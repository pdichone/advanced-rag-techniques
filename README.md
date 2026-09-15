<!-- @format -->
# Advanced RAG Techniques — Course Code

Source code for the **Advanced RAG Techniques** course on Udemy by Paulo Dichone. Every script here is
one you see built on screen. Run them in the order the course teaches them, against the same document
the course uses, and you will get the same results you see in the lectures.

**What the course covers:** you start with a naive RAG pipeline, watch it fail, and then fix it with
one technique at a time. Query expansion. Re-ranking with a cross-encoder. Dense passage retrieval.
Each technique gets its own script, and each script is self-contained so you can run it on its own.

## Setup

```bash
git clone https://github.com/pdichone/advanced-rag-techniques.git
cd advanced-rag-techniques
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

The first run of `reranking.py` and `dpr_technique.py` downloads models from Hugging Face. Expect a
few minutes and a couple of GB the first time; after that they load from cache.

## Which script goes with which section

| Course section | Script | What it does |
|---|---|---|
| 4 · Query Expansion with Generated Answers | `expansion_answer.py` | Asks the LLM for a hypothetical answer, appends it to the query, and retrieves with the joint query. Plots the original vs augmented query in embedding space |
| 5 · Query Expansion with Multiple Queries | `expansion_queries.py` | Asks the LLM for five related questions, retrieves for all of them, and de-duplicates the union |
| 6 · Re-Ranking with a Cross-Encoder | `reranking.py` | Retrieves a wide candidate set, then re-scores every candidate with `cross-encoder/ms-marco-MiniLM-L-6-v2` before generating the answer |
| 7 · Dense Passage Retrieval | `dpr_technique.py` | Encodes questions and passages with Facebook's DPR question and context encoders and ranks by cosine similarity |
| shared | `helper_utils.py` | PDF text extraction, Chroma loading, text wrapping, UMAP projection for the embedding plots |

Sections 1 to 3 (setup, RAG fundamentals, the RAG triad, naive RAG and its drawbacks) are concept
lectures and have no separate script. Section 8 (contextual retrieval, late chunking, agentic RAG,
GraphRAG, multimodal RAG) ships its code as downloadable resources attached to those lectures.

## The data

`data/microsoft-annual-report.pdf` is the one document every script indexes. Using the same corpus
throughout is deliberate: when a technique changes what gets retrieved, you know the technique did
it, not the data.

Each script re-reads the PDF, chunks it, and builds a fresh in-memory Chroma collection on every run.
There is nothing to clean up between runs.

## The stack

| Piece | What the code uses |
|---|---|
| Chat model | OpenAI `gpt-5.6-luna` (default argument on the generation functions; change it there) |
| Embeddings | Local `sentence-transformers` via Chroma's default embedding function. No API key needed for retrieval |
| Vector store | Chroma, in memory |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` from `sentence-transformers` |
| DPR | `facebook/dpr-question_encoder-single-nq-base` and `facebook/dpr-ctx_encoder-single-nq-base` from `transformers` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` then `SentenceTransformersTokenTextSplitter` |

**Model note, September 2026.** The scripts used to default to `gpt-3.5-turbo`, which OpenAI is
retiring on 23 October 2026. They now default to `gpt-5.6-luna`. If you cloned before then, pull
the latest. If OpenAI has moved on again by the time you read this, change the `model=` default in
the three generation functions and everything else keeps working.

## Running the scripts

```bash
python expansion_answer.py
python expansion_queries.py
python reranking.py
python dpr_technique.py
```

Each script prints the query, the retrieved chunks, and the generated answer so you can compare
the retrieval before and after the technique is applied. Lines that are commented out in the source
are the intermediate prints used on screen in the lectures; uncomment them to follow along.

## Troubleshooting

- **`OPENAI_API_KEY` not found.** The `.env` file must sit in the project root, next to the scripts, and you must run the scripts from that folder.
- **Slow first run.** Model downloads from Hugging Face. Subsequent runs are fast.
- **`torch` install trouble.** Install the CPU build from pytorch.org for your platform, then re-run `pip install -r requirements.txt`.
- **Different chunk counts than the video.** Library versions drift. The retrieval behaviour is what matters, not the exact number.

---

## Go further: The AI Guild

This code is part of a module in [The AI Guild](https://bit.ly/ai-guild-join), a community of
developers and entrepreneurs building real AI tools together. Weekly live calls, code and template
vault, guided learning paths from the basics through production systems, and direct feedback on
what you build. All skill levels welcome.
