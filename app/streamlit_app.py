"""Streamlit entry point for interacting with an OpenAI Agent Builder agent (no session_id)."""
from __future__ import annotations

import os
import uuid
import subprocess
from typing import Any, List, Dict

import streamlit as st
from openai import OpenAI, __version__ as openai_version


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
    msgs: List[Dict[str, Any]] = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        msgs.append({"role": role, "content": [{"type": "text", "text": content}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": new_user_text}]})
    return msgs


def _call_agent_minimal(client: OpenAI, agent_id: str, user_text: str) -> str:
    """오로지 필수 인자만(모델+입력)으로 테스트 호출"""
    resp = client.responses.create(
        model=agent_id,
        input=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        # ❌ 절대 session_id 넣지 않음
    )
    return _extract_text(resp)


def _call_agent_with_history(client: OpenAI, agent_id: str, messages: List[Dict[str, Any]]) -> str:
    """히스토리 누적 호출(역시 session_id 없음)"""
    resp = client.responses.create(model=agent_id, input=messages)
    return _extract_text(resp)


def _init_session_state() -> None:
    if "session_guid" not in st.session_state:
        # 로컬에서만 쓰는 식별자(서버로 안 보냄)
        st.session_state.session_guid = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []


def _grep_session_id_in_repo() -> str:
    """배포 환경에서 repo 내 session_id 흔적이 남았는지 간단 점검(옵션)"""
    try:
        out = subprocess.check_output(["bash", "-lc", "grep -RIn --line-number --ignore-case 'session_id' || true"])
        return out.decode("utf-8").strip()
    except Exception:
        return "(search failed or not available)"


def main() -> None:
    st.set_page_config(page_title="Agent Builder Chat", page_icon="🤖", layout="wide")
    st.title("OpenAI Agent Builder Chat (no session_id)")
    st.caption("Responses API는 session_id 인자를 받지 않습니다. 히스토리는 input에 누적해 전달하세요.")

    _init_session_state()

    client = _build_client()
    agent_id = os.getenv("OPENAI_AGENT_ID", "").strip()

    with st.sidebar:
        st.header("Diagnostics")
        st.write(f"OpenAI SDK: `{openai_version}`")
        st.write(f"OPENAI_AGENT_ID: `{agent_id or '(empty)'}`")
        st.write(f"Local session GUID: `{st.session_state.session_guid}`")

        # 레포지토리에서 session_id 흔적이 남았는지 검색(클라우드에서만 동작할 수 있음)
        if st.button("Scan repo for 'session_id'"):
            st.code(_grep_session_id_in_repo(), language="bash")

        st.header("Configuration")
        st.markdown(
            """
            Set environment variables:

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
        st.subheader("Minimal test call")
        test_text = st.text_input("Ping text", value="ping")
        if st.button("Run minimal call"):
            try:
                reply = _call_agent_minimal(client, agent_id, test_text)
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
