"""Streamlit entry point for interacting with an OpenAI Agent Builder agent."""
from __future__ import annotations

import os
import uuid
from typing import Any, List, Dict

import streamlit as st
from openai import OpenAI


def _build_client() -> OpenAI | None:
    """Initialise the OpenAI client if an API key is present."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _extract_text(response: Any) -> str:
    """Safely extract the assistant text from a responses.create payload."""
    if response is None:
        return ""

    # The latest OpenAI SDK exposes `output_text` for convenience.
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # Fallback for structured responses.
    text_chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", "")
                if text:
                    text_chunks.append(text)

    return "".join(text_chunks)


def _history_to_messages(history: List[Dict[str, str]], new_user_text: str) -> List[Dict[str, Any]]:
    """Convert local chat history + new user text into Responses API `input` format."""
    messages: List[Dict[str, Any]] = []
    for m in history:
        # keep only roles user/assistant and plain text content
        role = m.get("role", "user")
        content = m.get("content", "")
        messages.append(
            {
                "role": role,
                "content": [{"type": "text", "text": content}],
            }
        )
    # append the new user prompt
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": new_user_text}],
        }
    )
    return messages


def _call_agent(client: OpenAI, agent_id: str, messages: List[Dict[str, Any]]) -> str:
    """Send messages to the Agent Builder API and return the generated reply."""
    response = client.responses.create(
        model=agent_id,   # <- OPENAI_AGENT_ID 또는 Workflow/Agent ID
        input=messages,   # <- 누적 히스토리 + 새 유저 입력
        # ❌ session_id는 Responses.create에서 지원하지 않으므로 절대 넣지 말 것
    )
    return _extract_text(response)


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        # 로컬(클라이언트) 기준 세션 식별자 — API로는 보내지 않음
        st.session_state.session_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        # [{"role": "user"|"assistant", "content": "텍스트"}] 형태로 저장
        st.session_state.history = []


def main() -> None:
    st.set_page_config(page_title="Agent Builder Chat", page_icon="🤖", layout="wide")

    st.title("OpenAI Agent Builder Chat")
    st.caption(
        "Connect to an OpenAI Agent Builder agent to send prompts and view responses in real time."
    )

    _init_session_state()

    client = _build_client()
    agent_id = os.getenv("OPENAI_AGENT_ID", "").strip()

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            """
            Set the following environment variables in Streamlit Cloud (or locally) before
            launching the app:

            - `OPENAI_API_KEY`: your OpenAI API key with access to the Agent Builder.
            - `OPENAI_AGENT_ID`: the unique identifier of the agent you created in Agent Builder.
            """
        )

        if not client:
            st.error("Missing `OPENAI_API_KEY`. Set it in your environment to connect to OpenAI.")
        if not agent_id:
            st.warning("`OPENAI_AGENT_ID` is empty. Provide the agent ID to start chatting.")

        if st.button("Reset conversation", type="secondary"):
            st.session_state.history = []
            st.session_state.session_id = str(uuid.uuid4())
            try:
                st.rerun()  # 최신 스트림릿
            except Exception:
                st.experimental_rerun()  # 구버전 호환

    if client and agent_id:
        st.subheader("Chat with your agent")
        prompt = st.chat_input("Send a message to your agent…")

        # render previous messages
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt:
            # echo user message
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Waiting for the agent's response…"):
                        messages = _history_to_messages(st.session_state.history[:-1], prompt)
                        reply = _call_agent(client, agent_id, messages)
                        if not reply:
                            reply = "(The agent did not return any text.)"
                        st.markdown(reply)
            except Exception as exc:  # noqa: BLE001
                error_message = f"Failed to reach the Agent Builder API: {exc}"
                st.session_state.history.append({"role": "assistant", "content": error_message})
                st.error(error_message)
            else:
                st.session_state.history.append({"role": "assistant", "content": reply})
    else:
        st.info(
            "Add your OpenAI credentials to begin chatting with your Agent Builder agent."
        )


if __name__ == "__main__":
    main()
