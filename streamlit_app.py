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

import config
import rag

st.set_page_config(page_title="Document chatbot", page_icon="📄", layout="centered")


# ----------------------------------------------------------------- index
@st.cache_resource(show_spinner=False)
def db_client():
    """One Chroma client per session (reused across Streamlit reruns)."""
    return chromadb.PersistentClient(path=config.DB_DIR)


def index_stats():
    """(chunk_count, source filenames), or None when nothing has been ingested."""
    try:
        collection = db_client().get_collection(config.COLLECTION)
        count = collection.count()
    except Exception:
        return None
    if not count:
        return 0, []
    metas = collection.get(include=["metadatas"])["metadatas"]
    return count, sorted({m["source"] for m in metas})


def rebuild_index():
    """Run ingest.py in-process and return what it printed."""
    import ingest

    printed = io.StringIO()
    with redirect_stdout(printed):
        ingest.main()
    rag._collection = None          # the old collection object was deleted by ingest
    return printed.getvalue()


def chunk_texts(ids):
    """Chunk id -> chunk text, read straight from Chroma (no embedding needed)."""
    try:
        got = rag.get_collection().get(ids=ids, include=["documents"])
    except (Exception, SystemExit):
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
            try:
                st.session_state.ingest_log = rebuild_index()
            except BaseException as exc:          # ingest.py uses SystemExit for 'no docs'
                st.session_state.ingest_log = f"Failed: {exc}"

    stats = index_stats()
    has_index = bool(stats and stats[0])
    with status_slot.container():
        if not has_index:
            st.warning("No index yet.")
        else:
            st.metric("chunks indexed", stats[0])
            st.caption(" · ".join(stats[1]))

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
    st.info("Put `.md`, `.txt` or `.pdf` files in `docs/`, then press "
            "**Rebuild index** in the sidebar.")

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
            error = (f"Could not reach the model server at {config.OLLAMA_URL}.\n\n"
                     f"Start it with `ollama serve`, then ask again.\n\n`{exc}`")
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})
        else:
            st.markdown(result["answer"])
            render_details(result, show_debug)
            st.session_state.messages.append({"role": "assistant",
                                              "content": result["answer"],
                                              "result": result})
