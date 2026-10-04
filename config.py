"""
config.py - every setting in one place.
Change values here to run experiments (e.g. chunk size) without touching the logic.
"""
from pathlib import Path

# --- Files and folders ---
DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
COLLECTION = "docs"
LOG_FILE = Path("logs/rag_log.jsonl")
EVAL_LOG = Path("logs/eval_runs.jsonl")

OLLAMA_URL = "http://localhost:11434/v1"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

CHUNK_STRATEGY = "headings"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K = 4
MIN_SIMILARITY = 0.55         # below this, refuse without asking the LLM (tune with eval!)
MAX_ANSWER_TOKENS = 400
REFUSAL = "I don't know based on the provided documents."
