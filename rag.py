"""
rag.py -- the core: answer one question from the stored chunks.

question -> embed -> find top-k chunks in Chroma -> (refuse?) -> LLM -> log -> answer
"""
import json
from datetime import datetime, timezone

import chromadb

import config
from llm import chat, embed

SYSTEM_PROMPT = f"""You answer questions using ONLY the documents the user provides.
Rules:
- Use only facts stated in the documents. Never use outside knowledge.
- Cite the source file after each fact in square brackets, e.g. [refunds.md].
- If the documents do not contain the answer, reply with exactly: {config.REFUSAL}
- Keep answers short and direct."""

_collection = None


def get_collection():
    """Open the Chroma index once and reuse it."""
    global _collection
    if _collection is None:
        db = chromadb.PersistentClient(path=config.DB_DIR)
        try:
            _collection = db.get_collection(config.COLLECTION)
        except Exception:
            raise SystemExit("No index found. Run `python ingest.py` first.")
    return _collection


def retrieve(question):
    """Return the TOP_K most similar chunks, best first."""
    query_vector = embed([question], kind="query")[0]
    result = get_collection().query(query_embeddings=[query_vector], n_results=config.TOP_K)
    hits = []
    # Chroma returns lists of lists (one inner list per query); we sent one query -> [0]
    for chunk_id, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                               result["metadatas"][0], result["distances"][0]):
        hits.append({"id": chunk_id, "text": text, "source": meta["source"],
                     "page": meta["page"],
                     "similarity": 1 - distance})   # Chroma gives cosine *distance*
    return hits


def build_user_message(question, hits):
    docs = "\n".join(
        f'<document source="{h["source"]}" page="{h["page"]}">\n{h["text"]}\n</document>'
        for h in hits)
    return f"<documents>\n{docs}\n</documents>\n\n<question>{question}</question>"


def log(record):
    config.LOG_FILE.parent.mkdir(exist_ok=True)
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def answer(question):
    """Answer one question. Returns a dict with the answer, sources and details."""
    hits = retrieve(question)
    best = max((h["similarity"] for h in hits), default=0.0)

    if best < config.MIN_SIMILARITY:
        # Guard 1: nothing relevant was found, so don't even ask the LLM.
        text, finish, usage, stage = config.REFUSAL, "below_threshold", None, "retrieval"
    else:
        # Guard 2: the system prompt tells the LLM to refuse if the chunks don't answer it.
        text, finish, usage = chat(SYSTEM_PROMPT, build_user_message(question, hits))
        stage = "llm"

    refused = "i don't know" in text.lower()

    result = {
        "question": question,
        "answer": text,
        "refused": refused,
        "refused_at": stage if refused else None,
        "best_similarity": round(best, 4),
        "sources": [{"id": h["id"], "source": h["source"], "page": h["page"],
                     "similarity": round(h["similarity"], 4)} for h in hits],
        "finish_reason": finish,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "settings": {"model": config.CHAT_MODEL, "top_k": config.TOP_K,
                     "chunking": config.CHUNK_STRATEGY, "chunk_size": config.CHUNK_SIZE,
                     "min_similarity": config.MIN_SIMILARITY},
    }
    log(result)
    return result
