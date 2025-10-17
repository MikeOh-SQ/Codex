"""Streamlit entry point for interacting with an OpenAI Agent Builder agent (fixed content types)."""
from __future__ import annotations

import os
import uuid
from typing import Any, List, Dict

import streamlit as st
from openai import OpenAI


def _build_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _extract_text(response: Any) -> str:
    if response is None:
        return ""

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                t = getattr(c, "text", "")
                if t:
                    chunks.append(t)
    return "".join(chunks)


def _history_to_messages(history: List[Dict[str, str]], new_user_text: str) -> List[Dict[str, Any]]:
    """
    Responses API 요구 형식:
      - user 메시지:  type="input_text"
      - assistant 메시지: type="output_text"
    """
    msgs: List[Dict[str, Any]] = []
    for m in history:
        role = m.get("role", "user")
        text = m.get("content", "")
        if role == "assistant":
            msgs.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
        else:
            msgs.append({"role": "user", "content": [{"type": "input_text", "text": text}]})

    # 새 유저 입력
    msgs.append({"role": "user", "content": [{"type": "input_text", "text": new_user_text}]})
    return msgs


def _call_agent_minimal(client: OpenAI, agent_id: str, user_text: str) -> str:
    """필수 인자만으로 간단 호출 (콘텐츠 타입 고정)"""
    resp = client.responses.create(
        model="wf_68f0b0ef88088190abb547e69c16dfae0c8fcfb76c043f19",
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_text}]}],
    )
    return _extract_text(resp)


def _call_agent_with_history(client: OpenAI, agent_id: str, messages: List[Dict[str, Any]]) -> str:
    resp = client.responses.create(model="wf_68f0b0ef88088190abb547e69c16dfae0c8fcfb76c043f19", input=messages)
    return _extract_text(resp)


def _init_session_state() -> None:
    if "session_guid" not in st.session_state:
        st.session_state.session_guid = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []


def main() -> None:
    st.set_page_config(page_title="Agent Builder Chat", page_icon="🤖", layout="wide")

    st.title("OpenAI Agent Builder Chat")
    st.caption("Responses API: use input_text/output_text content types (no session_id).")

    _init_session_state()

    client = _build_client()
    agent_id = os.getenv("OPENAI_AGENT_ID", "").strip()

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            """
            Environment variables:
            - `OPENAI_API_KEY`
            - `OPENAI_AGENT_ID` (Agent/Workflow ID)
            """
        )

        if not client:
            st.error("Missing `OPENAI_API_KEY`.")
        if not agent_id:
            st.warning("`OPENAI_AGENT_ID` is empty.")

        if st.button("Reset conversation", type="secondary"):
            st.session_state.history = []
            st.session_state.session_guid = str(uuid.uuid4())
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

        st.divider()
        st.subheader("Minimal test")
        if st.button("Ping agent"):
            try:
                reply = _call_agent_minimal(client, agent_id, "ping")
                st.success("Minimal call OK")
                st.code(reply or "(no text)")
            except Exception as e:
                st.error(f"Minimal call failed: {e}")

    if not (client and agent_id):
        st.info("Add credentials to chat.")
        return

    st.subheader("Chat with your agent")
    prompt = st.chat_input("Send a message to your agent…")

    # render history
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Waiting for the agent's response…"):
                    messages = _history_to_messages(st.session_state.history[:-1], prompt)
                    reply = _call_agent_with_history(client, agent_id, messages)
                    if not reply:
                        reply = "(The agent did not return any text.)"
                    st.markdown(reply)
        except Exception as exc:
            err = f"Failed to reach the Agent Builder API: {exc}"
            st.session_state.history.append({"role": "assistant", "content": err})
            st.error(err)
        else:
            st.session_state.history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
