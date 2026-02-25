import json
import streamlit as st
from src.ui.api_client import client, stream_client, api_url

st.set_page_config(page_title="Agent", layout="wide")
st.title("Agent")
st.caption("Chat with the agentic RAG (streams via SSE)")

# --- Session management ---
with st.sidebar:
    st.header("Session")
    collection_name = st.text_input("Collection name", value="research_papers")
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)

    if st.button("New session"):
        try:
            resp = client.post(
                api_url("/sessions"),
                json={"collection_name": collection_name, "top_k": top_k},
            )
            if resp.status_code == 201:
                st.session_state["session_id"] = resp.json()["session_id"]
                st.session_state["chat_history"] = []
                st.success(f"Session: {st.session_state['session_id'][:8]}…")
                st.rerun()
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")

    if "session_id" in st.session_state:
        st.info(f"Active: `{st.session_state['session_id'][:8]}…`")
        if st.button("Delete session"):
            try:
                resp = client.delete(api_url(f"/sessions/{st.session_state['session_id']}"))
                if resp.status_code == 200:
                    del st.session_state["session_id"]
                    st.session_state["chat_history"] = []
                    st.rerun()
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")

# --- Chat history ---
chat_history = st.session_state.get("chat_history", [])
for turn in chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("contexts"):
            with st.expander("Contexts"):
                for ctx in turn["contexts"]:
                    st.markdown(f"- {ctx}")
        if turn["role"] == "assistant" and turn.get("citations"):
            with st.expander("Citations"):
                for cite in turn["citations"]:
                    st.markdown(f"- `{cite}`")

# --- Chat input ---
has_session = "session_id" in st.session_state
user_input = st.chat_input("Ask a question…", disabled=not has_session)

if user_input and has_session:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    chat_history.append({"role": "user", "content": user_input})

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        contexts: list[str] = []
        citations: list[str] = []

        try:
            with stream_client.stream(
                "POST",
                api_url("/chat/stream"),
                json={"session_id": st.session_state["session_id"], "message": user_input},
            ) as resp:
                event_type = None
                for line in resp.iter_lines():
                    if line.startswith("event:"):
                        event_type = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if event_type == "delta":
                            full_text += data.get("text", "")
                            placeholder.markdown(full_text + "▌")
                        elif event_type == "done":
                            contexts = data.get("contexts", [])
                            citations = data.get("citations", [])
                        elif event_type == "error":
                            st.error(data.get("detail", "Unknown error"))
                            break

            placeholder.markdown(full_text)

            if contexts:
                with st.expander("Contexts"):
                    for ctx in contexts:
                        st.markdown(f"- {ctx}")
            if citations:
                with st.expander("Citations"):
                    for cite in citations:
                        st.markdown(f"- `{cite}`")

        except Exception as exc:
            st.error(f"Stream error: {exc}")
            full_text = f"Error: {exc}"

    chat_history.append(
        {"role": "assistant", "content": full_text, "contexts": contexts, "citations": citations}
    )
    st.session_state["chat_history"] = chat_history
