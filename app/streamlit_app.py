# app.py — Streamlit x OpenAI Agents SDK (Runner 방식)
from __future__ import annotations
import os
import uuid
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# Agents SDK
from agents import Agent, Runner, SQLiteSession
from agents import set_default_openai_client, set_default_openai_key
from openai import AsyncOpenAI

load_dotenv()  # .env 사용 시

# ---- 기본 설정 ----
APP_TITLE = "Agent Builder (Agents SDK) Chat"
WORKFLOW_ID = os.getenv("OPENAI_AGENT_ID") or os.getenv("OPENAI_WORKFLOW_ID")  # wf_/agt_ 모두 허용
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # proj_...

# OpenAI 클라이언트(프로젝트 강제 지정 — 키와 같은 프로젝트여야 함)
if OPENAI_API_KEY:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT)
    set_default_openai_client(client)
    set_default_openai_key(OPENAI_API_KEY)

# ---- Streamlit UI ----
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.title(APP_TITLE)
st.caption("Agents SDK의 Runner로 워크플로(또는 에이전트)를 실행합니다. (session_id 미사용)")

# 세션 상태 초기화
if "chat_session" not in st.session_state:
    # SQLiteSession: 자동으로 히스토리 저장/복원
    st.session_state.chat_session = SQLiteSession(session_id=str(uuid.uuid4()))
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

# 사이드바: 디버그/설정
with st.sidebar:
    st.header("Configuration")
    st.write("**OPENAI_API_KEY set?**", bool(OPENAI_API_KEY))
    st.write("**OPENAI_PROJECT**", OPENAI_PROJECT or "(empty)")
    st.write("**WORKFLOW/AGENT ID**", WORKFLOW_ID or "(empty)")

    st.markdown(
        """
        **필수 조건**
        1) `OPENAI_API_KEY`는 **워크플로가 속한 같은 프로젝트**에서 발급  
        2) `OPENAI_PROJECT=proj_...` 도 같은 프로젝트  
        3) `OPENAI_AGENT_ID` 는 `wf_...` 또는 `agt_...` 그대로
        """
    )

    if st.button("Reset conversation", type="secondary"):
        st.session_state.chat_session = SQLiteSession(session_id=str(uuid.uuid4()))
        st.session_state.history = []
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

# 가드: 인증/ID 미설정 시 안내
if not OPENAI_API_KEY or not WORKFLOW_ID:
    st.info("환경변수 `OPENAI_API_KEY`, `OPENAI_PROJECT`, `OPENAI_AGENT_ID(wf_/agt_)`를 설정해 주세요.")
    st.stop()

# 에이전트 정의(Builder의 예시와 유사)
base_agent = Agent(
    name="My agent",
    instructions="You are a helpful assistant.",
    model="gpt-5",  # Builder 코드와 동일한 기본 모델명
    model_settings={"reasoning": {"effort": "low", "summary": "auto"}, "store": True},
)

# 채팅 UI
st.subheader("Chat")
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("메시지를 입력하세요…")

def run_agent(user_text: str) -> str:
    """
    Agents SDK Runner로 워크플로 실행.
    - 대화 히스토리는 SQLiteSession이 자동 관리
    - workflow_id는 trace_metadata로 전달 (Agent Builder 'Get code' 예시와 동일 패턴)
    """
    input_items = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }
    ]
    result = Runner.run_sync(
        base_agent,
        input_items,
        session=st.session_state.chat_session,
        run_config={
            "trace_metadata": {
                "__trace_source__": "agent-builder",
                "workflow_id": WORKFLOW_ID,  # ★ 핵심: 워크플로 식별자 전달
            },
            # 필요시 공통 모델/세팅 오버라이드:
            # "model": "gpt-5-mini",
            # "model_settings": {"temperature": 0.2},
        },
    )
    # 최종 텍스트(없을 땐 빈 문자열)
    return (result.final_output or "").strip()

if prompt:
    # 사용자 메시지 표시/저장
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Agent 실행 중…"):
                reply = run_agent(prompt)
                if not reply:
                    reply = "(no output_text)"
                st.markdown(reply)
    except Exception as e:
        err = f"Agent run failed: {e}"
        st.session_state.history.append({"role": "assistant", "content": err})
        st.error(err)
    else:
        st.session_state.history.append({"role": "assistant", "content": reply})
