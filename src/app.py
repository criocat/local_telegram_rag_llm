from __future__ import annotations

import uuid
import streamlit as st

from src.retrieval import build_engine

st.set_page_config(page_title="Local Telegram RAG", layout="wide")

if "engine" not in st.session_state:
    st.session_state.engine = build_engine()
if "chats" not in st.session_state:
    first_chat_id = str(uuid.uuid4())
    st.session_state.chats = {first_chat_id: {"name": "New Chat", "messages": []}}
    st.session_state.current_chat = first_chat_id
if "debug" not in st.session_state:
    st.session_state.debug = {}

with st.sidebar:
    st.title("Chats")
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"name": f"Chat {len(st.session_state.chats) + 1}", "messages": []}
        st.session_state.current_chat = new_id
        st.session_state.debug = {}
        st.rerun()
        
    st.divider()
    
    for chat_id, chat_data in list(st.session_state.chats.items()):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(chat_data["name"], key=f"btn_{chat_id}", use_container_width=True, disabled=(chat_id == st.session_state.current_chat)):
                st.session_state.current_chat = chat_id
                st.session_state.debug = {}
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{chat_id}"):
                del st.session_state.chats[chat_id]
                if st.session_state.current_chat == chat_id:
                    if st.session_state.chats:
                        st.session_state.current_chat = list(st.session_state.chats.keys())[0]
                    else:
                        new_id = str(uuid.uuid4())
                        st.session_state.chats[new_id] = {"name": "New Chat", "messages": []}
                        st.session_state.current_chat = new_id
                st.session_state.debug = {}
                st.rerun()

st.title("Local Telegram RAG")

chat_col, debug_col = st.columns([2, 1])

current_messages = st.session_state.chats[st.session_state.current_chat]["messages"]

with chat_col:
    st.subheader(st.session_state.chats[st.session_state.current_chat]["name"])

    # Add a container for all chat messages so they scroll together cleanly
    chat_container = st.container()
    
    with chat_container:
        for msg in current_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_input := st.chat_input("Ask about your Telegram history"):
        # Auto-rename the chat based on the first user message
        if st.session_state.chats[st.session_state.current_chat]["name"] == "New Chat" or st.session_state.chats[st.session_state.current_chat]["name"].startswith("Chat "):
            new_name = user_input[:20] + ("..." if len(user_input) > 20 else "")
            st.session_state.chats[st.session_state.current_chat]["name"] = new_name

        # Snapshot history BEFORE appending the new user message, to pass to the engine
        history_for_llm = [{"role": m["role"], "content": m["content"]} for m in current_messages]

        current_messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        result = st.session_state.engine.retrieve(user_input, chat_history=history_for_llm)
        
        st.session_state.debug = {
            "search_query": result.router.search_query,
            "requires_history": result.router.requires_history,
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

        with chat_container:
            with st.chat_message("assistant"):
                answer_stream = st.session_state.engine.llm.generate_answer_stream(
                    question=user_input,
                    expanded_context=result.expanded_context,
                    chat_history=history_for_llm
                )
                answer = st.write_stream(answer_stream)
                
        current_messages.append({"role": "assistant", "content": answer})

with debug_col:
    st.subheader("Debug / Inspector")
    st.markdown("**Search Query**")
    st.info(st.session_state.debug.get("search_query", "None"))
    st.markdown("**Requires History**")
    st.info(str(st.session_state.debug.get("requires_history", True)))
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