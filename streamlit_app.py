"""
streamlit_app.py -- browser UI over the same core the CLI uses.

Usage:  streamlit run streamlit_app.py

This file only draws things. Every answer still comes from rag.answer(),
exactly as in app.py, so the UI and the CLI can never drift apart.
"""
import io
import json
from contextlib import redirect_stdout

import chromadb
import streamlit as st
from chromadb.errors import NotFoundError
from openai import APIConnectionError

import config
import rag

st.set_page_config(page_title="Document chatbot", page_icon="📄", layout="centered")


# ----------------------------------------------------------------- index
@st.cache_resource(show_spinner=False)
def db_client():
    """One Chroma client per session (reused across Streamlit reruns)."""
    return chromadb.PersistentClient(path=config.DB_DIR)


def index_status():
    """("ok"|"missing"|"empty"|"error", chunk count, source filenames, message).

    A failure to *read* the index is reported as itself. Reporting it as "no index"
    would blame docs/ for something that has nothing to do with the files there.
    """
    try:
        collection = db_client().get_collection(config.COLLECTION)
        count = collection.count()
        if not count:
            return "empty", 0, [], ""
        metas = collection.get(include=["metadatas"])["metadatas"]
        return "ok", count, sorted({m["source"] for m in metas}), ""
    except NotFoundError:
        return "missing", 0, [], ""
    except Exception as exc:
        return "error", 0, [], f"{type(exc).__name__}: {exc}"


def explain(exc):
    """Readable text for the usual failure: the model server is not running."""
    if isinstance(exc, APIConnectionError):
        return (f"Could not reach the model server at {config.OLLAMA_URL}. "
                f"Start it with `ollama serve`, then try again.")
    return f"{type(exc).__name__}: {exc}"


def rebuild_index():
    """Run ingest.py in-process. Returns (what it printed, error message or None).

    Embedding happens before ingest.py touches Chroma, so a failure here leaves
    the existing index exactly as it was.
    """
    import ingest

    printed = io.StringIO()
    try:
        with redirect_stdout(printed):
            ingest.main()
    except BaseException as exc:          # ingest.py uses SystemExit for 'no docs'
        return printed.getvalue(), explain(exc)
    rag._collection = None                # the old collection object was deleted by ingest
    return printed.getvalue(), None


def doc_files():
    """The files ingest.py would pick up, so the UI can say what it can see."""
    try:
        return sorted(p.name for p in config.DOCS_DIR.iterdir()
                      if p.suffix.lower() in (".md", ".txt", ".pdf"))
    except OSError:
        return []


def chunk_texts(ids):
    """Chunk id -> chunk text, read straight from Chroma (no embedding needed)."""
    try:
        got = db_client().get_collection(config.COLLECTION).get(ids=ids,
                                                                include=["documents"])
    except Exception:
        return {}
    return dict(zip(got["ids"], got["documents"]))


def recent_log(limit=10):
    try:
        lines = config.LOG_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-limit:][::-1]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({"time": entry.get("ts", "")[11:19],
                     "question": entry.get("question", ""),
                     "refused": entry.get("refused"),
                     "similarity": entry.get("best_similarity")})
    return rows


# ---------------------------------------------------------------- drawing
def render_details(result, show_debug):
    """The caption under an answer, plus the retrieval panel when debug is on."""
    if result["refused"]:
        st.caption(f"Refused at the *{result['refused_at']}* stage "
                   f"(best similarity {result['best_similarity']}).")
    else:
        files = sorted({s["source"] for s in result["sources"]})
        st.caption("searched: " + ", ".join(files))

    if not show_debug:
        return

    with st.expander(f"Retrieved {len(result['sources'])} chunks · "
                     f"best similarity {result['best_similarity']}"):
        st.caption(f"tokens in/out {result['input_tokens']}/{result['output_tokens']} · "
                   f"finish: {result['finish_reason']} · "
                   f"top_k {result['settings']['top_k']} · "
                   f"threshold {result['settings']['min_similarity']} · "
                   f"model {result['settings']['model']}")
        texts = chunk_texts([s["id"] for s in result["sources"]])
        for hit in result["sources"]:
            page = f" · page {hit['page']}" if hit["page"] else ""
            st.markdown(f"**{hit['similarity']:.3f}** · `{hit['id']}`{page}")
            st.progress(min(max(hit["similarity"], 0.0), 1.0))
            body = texts.get(hit["id"], "(chunk no longer in the index)")
            st.text(body[:1200] + ("…" if len(body) > 1200 else ""))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Index")
    # The status is drawn into this slot further down, after a rebuild may have
    # changed it. Rebuilding never calls st.rerun(): a run cut short that way
    # would drop the widget state of everything below it (the debug toggle).
    status_slot = st.empty()

    if st.button("Rebuild index", use_container_width=True,
                 help="Re-reads docs/, re-chunks and re-embeds everything."):
        with st.spinner("Embedding chunks — this is the slow part…"):
            st.session_state.ingest_log, st.session_state.ingest_error = rebuild_index()

    state, count, files, message = index_status()
    has_index = state == "ok"
    with status_slot.container():
        if has_index:
            st.metric("chunks indexed", count)
            st.caption(" · ".join(files))
        elif state == "error":
            st.error(f"Could not read the index: {message}")
        else:
            st.warning("No index yet." if state == "missing" else "The index is empty.")

    if st.session_state.get("ingest_error"):
        st.error(st.session_state.ingest_error)
    if st.session_state.get("ingest_log"):
        st.code(st.session_state.ingest_log.strip(), language=None)

    st.divider()
    st.subheader("Retrieval")
    config.TOP_K = st.slider("Chunks retrieved (top_k)", 1, 10, config.TOP_K, key="top_k")
    config.MIN_SIMILARITY = st.slider("Refuse below similarity", 0.0, 1.0,
                                      config.MIN_SIMILARITY, 0.01, key="min_sim",
                                      help="Below this, the question is refused without "
                                           "asking the model at all.")
    # Keys keep widget state tied to a name rather than to a position in the layout.
    show_debug = st.toggle("Show retrieved chunks", value=False, key="show_debug",
                           help="Same as :debug in the CLI.")
    st.caption(f"chat `{config.CHAT_MODEL}` · embed `{config.EMBED_MODEL}`\n\n"
               f"via {config.OLLAMA_URL}")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

    with st.expander("Recent queries (log)"):
        rows = recent_log()
        if rows:
            st.dataframe(rows, hide_index=True)
        else:
            st.caption("No queries logged yet.")

# ------------------------------------------------------------------- chat
st.title("📄 Document chatbot")
st.caption("Answers come only from the files in `docs/`, with the source cited. "
           "If the documents don't cover it, the bot says so.")

st.session_state.setdefault("messages", [])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("result"):
            render_details(message["result"], show_debug)

if not has_index:
    found = doc_files()
    if found:
        # The files are there; they just have not been embedded yet. Say that,
        # rather than telling someone to add files they can see in the folder.
        st.info(f"`docs/` has {len(found)} file(s) ready to index "
                f"({', '.join(found)}). Press **Rebuild index** in the sidebar.")
    else:
        st.info(f"No `.md`, `.txt` or `.pdf` files in `{config.DOCS_DIR}/`. "
                f"Add some, then press **Rebuild index** in the sidebar.")

question = st.chat_input("Ask about your documents", disabled=not has_index)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents…"):
                result = rag.answer(question)
        except Exception as exc:
            error = explain(exc)
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})
        else:
            st.markdown(result["answer"])
            render_details(result, show_debug)
            st.session_state.messages.append({"role": "assistant",
                                              "content": result["answer"],
                                              "result": result})
