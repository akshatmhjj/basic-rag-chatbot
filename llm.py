"""
llm.py -- the only file that talks to the model server.
Both ingest.py and rag.py import from here
later means changing config.py, not every file.
"""
from openai import OpenAI

import config

client = OpenAI(base_url=config.OLLAMA_URL, api_key="ollama")  # Ollama ignores the key


def embed(texts, kind):
    prefix = "search_document: " if kind == "document" else "search_query: "
    vectors = []
    for i in range(0, len(texts), 32):      # send 32 texts per request
        batch = [prefix + t for t in texts[i:i + 32]]
        resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def chat(system_prompt, user_message):
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        max_tokens=config.MAX_ANSWER_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    choice = resp.choices[0]
    return choice.message.content.strip(), choice.finish_reason, resp.usage
