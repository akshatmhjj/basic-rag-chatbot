# RAG Document Chatbot

A chatbot that answers questions **only** from your own documents (Markdown, text and PDF),
cites its sources, and refuses when the documents don't contain the answer.
Built from scratch (no LangChain), running fully free and local with Ollama and Chroma.

## How it works

```
                 INGEST (python ingest.py, run once)
docs/*.md, *.pdf ──► load ──► chunk ──► embed ──► Chroma (chroma_db/)

                 ANSWER (python app.py, every question)
question ──► embed ──► top-k similar chunks ──► similarity < threshold? ──► refuse
                                     │
                                     └──► LLM with "answer only from these docs" ──► answer + citations
                                                                                       │
                                                                     logs/rag_log.jsonl ◄┘
```

| File | Job |
|---|---|
| `config.py` | Every setting (models, chunk size, top-k, threshold) |
| `llm.py` | Talks to Ollama: `embed()` and `chat()` |
| `ingest.py` | Load files, chunk, embed, store in Chroma |
| `rag.py` | Retrieve, refuse or generate, log |
| `app.py` | Command-line chat |
| `eval.py` + `eval_set.jsonl` | Scores the bot on fixed test questions |

## Setup

```bash
# 1. Models (free, local)
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Build the index, then chat
python ingest.py
python app.py
```

Run `python eval.py` to score the bot. Re-run `python ingest.py` after changing documents
or any chunking setting.

## Design decisions

### Refusal: two guards
1. **Retrieval guard:** if the best chunk's cosine similarity is below `MIN_SIMILARITY`,
   the bot refuses without calling the LLM (cheaper, and immune to the model ignoring rules).
2. **Prompt guard:** the system prompt requires an exact refusal sentence when the
   retrieved chunks don't answer the question.

### Chunking strategy
_Fill this in with your own eval numbers (from `logs/eval_runs.jsonl`)._

| Strategy | Chunk size | Retrieval hit rate | Answer fact rate | Avg input tokens |
|---|---|---|---|---|
| fixed | 300 | | | |
| fixed | 1000 | | | |
| headings | 800 max | | | |

**Choice:** _..._ because _..._

### Threshold
`MIN_SIMILARITY` was chosen by comparing best-similarity scores for answerable vs
unanswerable questions (see `:debug` mode in `app.py`): _fill in the values you observed_.

## Evaluation
`eval.py` runs every question in `eval_set.jsonl` and reports:
- **retrieval_hit_rate**: the expected source file was among the retrieved chunks
- **answer_fact_rate**: the answer contains the required facts
- **refusal_accuracy**: refused unanswerable questions *and* answered answerable ones

## Limitations / next steps
- Scanned PDFs have no text layer and are skipped (would need OCR).
- Dense retrieval only; hybrid search (BM25 + embeddings) is the next improvement.
- Small local model can ignore instructions; the retrieval guard compensates.
