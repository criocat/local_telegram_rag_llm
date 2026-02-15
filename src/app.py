from __future__ import annotations

import streamlit as st

from src.retrieval import build_engine


st.set_page_config(page_title="Local Telegram RAG", layout="wide")
st.title("Local Telegram RAG")

if "engine" not in st.session_state:
    st.session_state.engine = build_engine()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug" not in st.session_state:
    st.session_state.debug = {}

chat_col, debug_col = st.columns([2, 1])

with chat_col:
    st.subheader("Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your Telegram history")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        result = st.session_state.engine.retrieve(user_input)
        answer = st.session_state.engine.llm.generate_answer(
            question=user_input,
            expanded_context=result.expanded_context,
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.debug = {
            "router_filters": result.router.filters,
            "chunks": [
                {
                    "point_id": c.point_id,
                    "score": c.score,
                    "author": c.payload.get("author_name"),
                    "text": c.payload.get("text"),
                }
                for c in result.top_chunks
            ],
            "expanded_context": result.expanded_context,
        }
        with st.chat_message("assistant"):
            st.markdown(answer)

with debug_col:
    st.subheader("Debug / Inspector")
    st.markdown("**Router filters (JSON)**")
    st.json(st.session_state.debug.get("router_filters", {}))
    st.markdown("**Retrieved chunks**")
    for chunk in st.session_state.debug.get("chunks", []):
        st.write(
            f"ID: `{chunk['point_id']}` | score: `{chunk['score']:.4f}` | author: `{chunk['author']}`"
        )
        st.code(chunk["text"] or "", language="text")
    st.markdown("**Expanded context passed to LLM**")
    st.code(st.session_state.debug.get("expanded_context", ""), language="text")
