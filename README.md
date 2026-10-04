# RAG Document Chatbot

A chatbot that answers questions **only** from your own documents (Markdown, text and PDF),
cites its sources, and refuses when the documents don't contain the answer.

Built from scratch with no LangChain. Runs fully free and local with **Ollama** (LLM and
embeddings) and **Chroma** (vector database).

- Answers grounded in your documents, with source citations
- Two-layer refusal for off-topic questions
- Document-aware chunking (by markdown heading), PDFs split per page
- Every question and answer logged to JSONL
- Eval script with retrieval, answer and refusal metrics

---

## Quick start: run order

| Step | Command | What it does | How often |
|---|---|---|---|
| 0 | `ollama serve` | Starts the local model server (leave it running) | Every session |
| 1 | `ollama pull llama3.2` and `ollama pull nomic-embed-text` | Downloads the chat model and the embedding model | Once |
| 2 | `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` | Creates the Python environment and installs packages | Once |
| 3 | `python ingest.py` | Reads `docs/`, chunks, embeds and saves to `chroma_db/` | Once, and again whenever docs or chunk settings change |
| 4 | `python app.py` | Opens the chat | Whenever you want to ask questions |
| 5 | `python eval.py` | Scores the bot on `eval_set.jsonl` | After every change you want to measure |

Inside the chat, type `:debug` to show similarity scores and retrieved chunks, and `:quit` to exit.

---

## How it works

The system has two separate flows. Ingest is slow and runs rarely; answering is fast and
runs on every question.

```
INGEST  (python ingest.py)
  docs/*.md, *.txt, *.pdf
        │
        ▼
  1. Load      .md/.txt read whole · PDFs read page by page
        ▼
  2. Chunk     split markdown at each # heading (PDFs: fixed-size pieces)
        ▼
  3. Embed     each chunk → vector via nomic-embed-text
        ▼
  4. Store     vectors + text + source/page saved in Chroma (chroma_db/)


ANSWER  (python app.py → rag.answer())
  question
        │
        ▼
  1. Embed the question
        ▼
  2. Retrieve the top 4 most similar chunks from Chroma
        ▼
  3. Guard 1: best similarity < 0.55? ──yes──► refuse (no LLM call)
        │ no
        ▼
  4. Generate: LLM gets the chunks + "answer ONLY from these, cite sources, else refuse"
        ▼
  5. Log question, chunks, scores, answer and tokens → logs/rag_log.jsonl
        ▼
  answer + citations
```

---

## Project structure

```
rag-chatbot/
├── docs/                 your documents (.md, .txt, .pdf)
├── config.py             all settings
├── llm.py                talks to Ollama
├── ingest.py             builds the index
├── rag.py                answers one question
├── app.py                command-line chat
├── eval.py               scores the bot
├── eval_set.jsonl        test questions
├── requirements.txt      openai, chromadb, pypdf
├── chroma_db/            created by ingest.py (git-ignored)
└── logs/                 created on first question (git-ignored)
```

### What each file does

**`config.py`** stores every setting: model names, chunk size, top-k, refusal threshold, file
paths. Experiments only change this file, never the logic.

**`llm.py`** is the only file that talks to the model server.
- `embed(texts, kind)` turns texts into vectors in batches of 32. It adds the
  `search_document:` / `search_query:` prefixes that nomic-embed-text was trained with.
- `chat(system, user)` sends one request at temperature 0 and returns the answer text,
  finish reason and token usage.

Switching to Gemini or Groq means changing `config.py`, because every other file goes through `llm.py`.

**`ingest.py`** builds the searchable index.
- `load_documents()` reads files; PDFs become one entry per page so answers can cite pages.
- `chunk_by_headings()` makes one chunk per markdown section, with the heading kept on every
  piece. A title with no text under it is merged into the next section.
- `chunk_fixed()` cuts every N characters with overlap; used for PDFs and for experiments.
- `main()` embeds all chunks and saves them to Chroma. It deletes the old collection first so
  stale chunks don't linger.

**`rag.py`** contains the core logic.
- `retrieve()` embeds the question and asks Chroma for the closest chunks. Chroma returns
  cosine *distance*, which is converted to similarity (`1 − distance`).
- `answer()` applies the threshold guard, calls the LLM, detects refusals and logs everything.
- `SYSTEM_PROMPT` holds the grounding rules: only use the documents, cite sources, and refuse
  with an exact sentence.

**`app.py`** is a `while` loop around `rag.answer()`. Debug mode prints the best similarity,
which guard refused, token counts and the IDs of the retrieved chunks.

**`eval.py`** runs every question in `eval_set.jsonl` through the real pipeline, prints
PASS/FAIL per question, prints a scorecard, details each failure, and appends a summary line
to `logs/eval_runs.jsonl` so runs with different settings can be compared.

---

## Key concepts

| Term | Meaning here |
|---|---|
| **Chunk** | A piece of a document small enough to retrieve precisely (here, one markdown section) |
| **Embedding** | A vector of numbers representing a text's meaning; similar meanings give nearby vectors |
| **Cosine similarity** | How closely two vectors point the same way; near 1 = similar meaning |
| **Top-k** | How many of the most similar chunks are sent to the LLM (4) |
| **Threshold** | Minimum similarity for the bot to even attempt an answer (0.55) |
| **Grounding** | Forcing the LLM to answer only from retrieved text, not its own memory |

---

## Configuration

| Setting | Value | Meaning |
|---|---|---|
| `CHAT_MODEL` | `llama3.2` | Model that writes answers |
| `EMBED_MODEL` | `nomic-embed-text` | Model that creates vectors |
| `CHUNK_STRATEGY` | `headings` | `headings` (by section) or `fixed` (every N characters) |
| `CHUNK_SIZE` | 800 | Max characters per chunk |
| `CHUNK_OVERLAP` | 100 | Characters shared between neighbouring fixed-size chunks |
| `TOP_K` | 4 | Chunks sent to the LLM per question |
| `MIN_SIMILARITY` | 0.55 | Below this, refuse without calling the LLM |
| `MAX_ANSWER_TOKENS` | 400 | Cap on answer length |

---

## Design decisions

### Refusal: two guards

1. **Retrieval guard (code):** if the best chunk's similarity is below `MIN_SIMILARITY`, the bot
   refuses without calling the LLM. It's cheap, and it can't be ignored the way a prompt
   instruction can.
2. **Prompt guard (LLM):** the system prompt requires an exact refusal sentence when the
   retrieved chunks don't answer the question. This handles borderline cases that pass the threshold.

Both are needed because, as the observed scores below show, no single threshold cleanly
separates answerable from unanswerable questions.

### Threshold

Best-similarity scores observed in `:debug` mode:

| Question | Best similarity | Should answer? | Result |
|---|---|---|---|
| How much do I need to pay for the Pro plan | 0.825 | Yes | Answered correctly |
| I bought a yearly plan 2 months ago. Can I get a refund? | 0.755 | Yes | Answered, but **wrong** (see Known failures) |
| (eval) I bought a yearly plan 10 days ago... | 0.726 | Yes | Wrongly refused by LLM |
| How many members can a team have | 0.666 | Yes | Partly correct |
| Can we get this plan for free | 0.619 | Ambiguous | Refused by LLM |
| `:exit` (typed as a question) | 0.593 | No | Refused by LLM |
| I lost my phone | 0.560 | Arguably (backup codes) | Refused by LLM |
| Can I pay with PhonePe? | 0.521 | No | Refused by threshold |

Clearly answerable questions scored **0.67–0.83**, and clearly unrelated ones scored around
**0.52**, with an overlap zone of about **0.56–0.62**. `0.55` refuses obvious off-topic questions
cheaply and leaves the overlap zone to the prompt guard. A higher threshold (around 0.60) would
start blocking borderline questions that might be answerable.

### Chunking strategy

Measured with `eval.py` on the 14-question eval set (llama3.2, top-k 4):

| Strategy | Chunk size | Retrieval hit rate | Answer fact rate | Refusal accuracy | Avg input tokens |
|---|---|---|---|---|---|
| fixed | 300 | _not run yet_ | | | |
| fixed | 1000 | _not run yet_ | | | |
| **headings** | **800 max** | **100%** | **90%** | **92.9%** | **331.6** |

To fill in the missing rows: set `CHUNK_STRATEGY = "fixed"` and `CHUNK_SIZE` in `config.py`, run
`python ingest.py`, then `python eval.py`. Copy the numbers from `logs/eval_runs.jsonl`.

**Choice: heading-based chunking.** Each section of these documents is one complete policy, for
example "Annual subscriptions" with both its 14-day rule and its prorated rule. Splitting by
heading keeps those related facts in a single chunk, and the heading travels with the text.
Retrieval found the expected source for every answerable question. The single answer failure
came from the LLM, not from retrieval (see below). _Confirm against the fixed-size rows once
they're filled in._

---

## Evaluation

`eval.py` runs each test question through the full pipeline and checks three things:

| Metric | Question it answers | Current |
|---|---|---|
| **retrieval_hit_rate** | Was the expected source file among the retrieved chunks? | 100% |
| **answer_fact_rate** | Does the answer contain the required facts (`must_contain`)? | 90% |
| **refusal_accuracy** | Refused unanswerable questions *and* answered answerable ones? | 92.9% |
| **avg_input_tokens** | Prompt size per question (cost and latency) | 331.6 |

Eval set: 14 questions, 10 answerable and 4 that should be refused. Result: **13/14 passed**.

Example test case:
```json
{"question": "How much does the Pro plan cost per year?", "answerable": true,
 "expected_source": "plans.md", "must_contain": ["2,990"]}
```

**Debugging rule:** check retrieval first. If retrieval hit the right source but the answer is
wrong, it's a generation problem (prompt or model). If retrieval missed, fix chunking or search first.

---

## Logging

Every question appends one line to `logs/rag_log.jsonl` with:
- the question and answer
- whether it refused, and which guard refused (`retrieval` or `llm`)
- best similarity and the retrieved chunk IDs with their scores
- finish reason and input/output token counts
- the settings used (model, top-k, chunking, threshold)

Each eval run appends a summary to `logs/eval_runs.jsonl`.

---

## Known failures

| Question | What happened | Cause |
|---|---|---|
| I bought a yearly plan 10 days ago. Can I get my money back? | Refused, even though the right chunk was retrieved (0.726) | **Generation:** the small model didn't connect "10 days" to "within 14 days" |
| I bought a yearly plan 2 months ago. Can I get a refund? | Claimed it was within the 14-day window and that the docs don't cover it, with no citation | **Generation:** the docs say "prorated minus 10% fee". A confident wrong answer, the most serious failure type |
| How many members can a team have? | Gave the minimum (3) as if answering the question | The docs state no maximum; the answer should have said so |
| Can we get this plan for free? | Refused | No conversation memory: "this plan" refers to an earlier question |

Planned fixes: add these cases to the eval set (with a `must_not_contain` check), add a
"quote the policy sentence first, then compare" rule to the prompt, and compare against a
larger model (`llama3.1:8b` or Gemini Flash) using the eval.

---

## Limitations and next steps

- **Small local model:** llama3.2 sometimes fails at reasoning over dates and numbers; the eval is set up to measure larger models.
- **No chat memory:** each question is independent; follow-ups need chat history and query rewriting.
- **Dense retrieval only:** hybrid search (BM25 + embeddings) would help with exact terms like error codes.
- **Scanned PDFs** have no text layer and are skipped (would need OCR).
- **CLI only:** a Streamlit UI is planned.