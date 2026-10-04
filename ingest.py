"""
ingest.py -- run this once, and again whenever your documents change.

load files -> split into chunks -> embed chunks -> save in Chroma
"""
import re

import chromadb
from pypdf import PdfReader

import config
from llm import embed


# 1. LOAD: read every file in docs/ as plain text
def load_documents():
    """Returns a list of {"source", "page", "text", "kind"}.
    PDFs become one entry per page so answers can cite page numbers."""
    documents = []
    for path in sorted(config.DOCS_DIR.iterdir()):
        suffix = path.suffix.lower()
        if suffix in (".md", ".txt"):
            documents.append({"source": path.name, "page": 0, "kind": "markdown",
                              "text": path.read_text(encoding="utf-8")})
        elif suffix == ".pdf":
            for page_number, page in enumerate(PdfReader(path).pages, start=1):
                text = page.extract_text() or ""     # scanned PDFs have no text layer
                if text.strip():
                    documents.append({"source": path.name, "page": page_number,
                                      "kind": "pdf", "text": text})
    return documents


# 2. CHUNK: two strategies, chosen in config.py
def chunk_fixed(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Cut every `size` characters. Neighbouring chunks share `overlap` characters,
    so a sentence cut at a boundary still appears whole in one of them."""
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def chunk_by_headings(text, max_size=config.CHUNK_SIZE):
    """Document-aware: one chunk per markdown section, heading included.
    A heading with no text under it (like the document title) is attached
    to the next section. Sections longer than max_size are split further."""
    sections = re.split(r"(?m)^(?=#{1,6} )", text)   # split just before each heading line
    chunks, carry = [], ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if "\n" not in section:                      # heading only, no body
            carry += section + "\n"
            continue
        section, carry = carry + section, ""
        if len(section) <= max_size:
            chunks.append(section)
        else:
            heading, _, body = section.partition("\n")
            for piece in chunk_fixed(body, max_size - len(heading) - 1):
                chunks.append(heading + "\n" + piece)   # every piece keeps its heading
    return chunks


def chunk_document(doc):
    # PDF text has no markdown headings, so PDFs always use fixed-size chunks.
    if doc["kind"] == "markdown" and config.CHUNK_STRATEGY == "headings":
        return chunk_by_headings(doc["text"])
    return chunk_fixed(doc["text"])


# 3 + 4. EMBED and STORE in Chroma
def main():
    documents = load_documents()
    if not documents:
        raise SystemExit(f"No .md, .txt or .pdf files found in {config.DOCS_DIR}/")

    ids, texts, metadatas = [], [], []
    for doc in documents:
        for i, piece in enumerate(chunk_document(doc)):
            ids.append(f"{doc['source']}-p{doc['page']}-c{i}")
            texts.append(piece)
            metadatas.append({"source": doc["source"], "page": doc["page"]})

    print(f"Loaded {len(documents)} documents/pages -> {len(texts)} chunks "
          f"(strategy={config.CHUNK_STRATEGY}, size={config.CHUNK_SIZE})")

    print("Embedding chunks (this is the slow part)...")
    vectors = embed(texts, kind="document")

    db = chromadb.PersistentClient(path=config.DB_DIR)
    try:
        db.delete_collection(config.COLLECTION)   # start fresh so old chunks don't linger
    except Exception:
        pass                                      # first run: nothing to delete
    collection = db.create_collection(config.COLLECTION,
                                      metadata={"hnsw:space": "cosine"})  # compare by cosine
    collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    print(f"Saved {collection.count()} chunks to ./{config.DB_DIR}/")


if __name__ == "__main__":
    main()
