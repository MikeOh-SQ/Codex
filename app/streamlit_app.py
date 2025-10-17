# streamlit_app.py — Agents SDK Runner + 트레이스/디버그 강화 버전

from __future__ import annotations
import os
import uuid
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# OpenAI / Agents SDK
from openai import OpenAI
from agents import Agent, Runner, set_default_openai_client

load_dotenv()

APP_TITLE = "Agent Builder (Agents SDK) Chat"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")                 # proj_...
WORKFLOW_OR_AGENT_ID = os.getenv("OPENAI_AGENT_ID") or os.getenv("OPENAI_WORKFLOW_ID")  # wf_/agt_
BASE_MODEL = os.getenv("OPENAI_BASE_MODEL", "gpt-4.1-mini")  # 접근 가능한 모델로

# OpenAI client (project 강제)
client: OpenAI | None = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT)
    set_default_openai_client(client)

st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.title(APP_TITLE)
st.caption("Agents SDK Runner로 워크플로/에이전트를 실행하고, Traces에서 실행 로그를 확인할 수 있도록 구성했습니다.")

# 세션 상태
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []
if "local_session_id" not in st.session_state:
    st.session_state.local_session_id = str(uuid.uuid4())

# 사이드바
with st.sidebar:
    st.header("Configuration")
    st.write("API key set?", bool(OPENAI_API_KEY))
    st.write("OPENAI_PROJECT:", OPENAI_PROJECT or "(empty)")
    st.write("WORKFLOW/AGENT ID:", WORKFLOW_OR_AGENT_ID or "(empty)")
    st.write("BASE_MODEL:", BASE_MODEL)

    st.markdown(
        """
        **필수 조건**
        1) API Key는 워크플로/에이전트와 **같은 프로젝트**에서 발급  
        2) `OPENAI_PROJECT=proj_...` 같은 프로젝트 지정  
        3) `OPENAI_AGENT_ID`: `wf_...` 또는 `agt_...`  
        4) 워크플로/에이전트는 **Publish/Deploy**
        """
    )

    if st.button("Reset conversation", type="secondary"):
        st.session_state.history = []
        st.session_state.local_session_id = str(uuid.uuid4())
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    st.divider()
    st.subheader("Quick Ping (Traces 확인용)")
    if st.button("Send ping"):
        try:
            # 아래 run_agent()와 동일 경로로 실행
            from datetime import datetime
            _ = None  # noqa

            # 간단 호출을 위해 내부에서 바로 실행
            base_agent = Agent(
                name="My agent",
                instructions="You are a helpful assistant.",
                model=BASE_MODEL,
            )
            runner = Runner()
            result = runner.run(
                base_agent,
                "ping",   # ✅ 문자열 입력 (안전)
                run_config={
                    "workflow_name": "Streamlit Chat (경민)",
                    "group_id": "streamlit-debug",
                    "trace_metadata": {
                        "__trace_source__": "agent-builder",
                        "workflow_id": WORKFLOW_OR_AGENT_ID,
                        "session": st.session_state.local_session_id,
                    },
                },
            )
            st.success("Ping sent. Check Traces dashboard.")
            st.code(repr(result)[:1200])  # 결과 미리보기
        except Exception as e:
            st.error(f"Ping failed: {e}")

# 가드
if not OPENAI_API_KEY or not WORKFLOW_OR_AGENT_ID:
    st.info("환경변수 `OPENAI_API_KEY`, `OPENAI_PROJECT`, `OPENAI_AGENT_ID(wf_/agt_)`를 설정해 주세요.")
    st.stop()

# Agent / Runner (전역)
base_agent = Agent(
    name="My agent",
    instructions="You are a helpful assistant.",
    model=BASE_MODEL,
)
runner = Runner()

def _fallback_parse_result(result: Any) -> str:
    # 1) final_output / finalOutput
    text = getattr(result, "final_output", None) or getattr(result, "finalOutput", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 2) new_items → assistant.output_text
    try:
        items = getattr(result, "new_items", []) or []
        chunks: List[str] = []
        for it in items:
            raw = getattr(it, "raw_item", None) or {}
            if raw.get("role") != "assistant":
                continue
            for c in raw.get("content", []) or []:
                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
        if chunks:
            return "\n".join(chunks).strip()
    except Exception:
        pass

    # 3) raw_responses
    try:
        raws = getattr(result, "raw_responses", []) or []
        for r in reversed(raws):
            txt = r.get("text") or r.get("output_text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    except Exception:
        pass

    return "(no output_text)"

def run_agent(user_text: str) -> Any:
    """Runner로 실행 + Traces 식별자 세팅."""
    result = runner.run(
        base_agent,
        user_text,  # ✅ 문자열 입력 (가장 단순/안전)
        run_config={
            "workflow_name": "Streamlit Chat (경민)",
            "group_id": "streamlit-debug",
            "trace_metadata": {
                "__trace_source__": "agent-builder",
                "workflow_id": WORKFLOW_OR_AGENT_ID,
                "session": st.session_state.local_session_id,
            },
        },
    )
    return result

# 디버그 토글
show_raw = st.sidebar.checkbox("Show raw run result")

# Chat UI
st.subheader("Chat")
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("메시지를 입력하세요…")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Agent 실행 중…"):
                result = run_agent(prompt)
                reply = _fallback_parse_result(result)
                st.markdown(reply)
                if show_raw:
                    st.sidebar.code(repr(result)[:2000])
    except Exception as e:
        err = f"Agent run failed: {e}"
        st.session_state.history.append({"role": "assistant", "content": err})
        st.error(err)
    else:
        st.session_state.history.append({"role": "assistant", "content": reply})
