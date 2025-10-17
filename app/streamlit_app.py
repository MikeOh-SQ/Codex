# streamlit_app.py — Streamlit x OpenAI Agents SDK (Runner 방식, 최소 옵션)

from __future__ import annotations
import os
import uuid
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# OpenAI / Agents SDK
from openai import OpenAI
from agents import Agent, Runner, set_default_openai_client

# -------------------- 기본 설정 --------------------
load_dotenv()  # .env 사용 시

APP_TITLE = "Agent Builder (Agents SDK) Chat"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # proj_...
WORKFLOW_OR_AGENT_ID = os.getenv("OPENAI_AGENT_ID") or os.getenv("OPENAI_WORKFLOW_ID")  # wf_... 또는 agt_...

# 모델은 계정에서 사용 가능한 것으로 지정(보통 gpt-4.1-mini 안전)
BASE_MODEL = os.getenv("OPENAI_BASE_MODEL", "gpt-4.1-mini")

# OpenAI 클라이언트(프로젝트 강제 지정; 키와 같은 프로젝트여야 함)
client: OpenAI | None = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT)
    set_default_openai_client(client)  # Agents SDK가 내부적으로 이 클라이언트를 사용

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.title(APP_TITLE)
st.caption("Agents SDK의 Runner로 워크플로우(또는 에이전트)를 실행합니다. Responses.create의 session_id는 사용하지 않습니다.")

# 세션 상태 초기화
if "history" not in st.session_state:
    # [{"role": "user"|"assistant", "content": "..."}, ...]
    st.session_state.history: List[Dict[str, str]] = []
if "local_session_id" not in st.session_state:
    st.session_state.local_session_id = str(uuid.uuid4())

# 사이드바: 설정/디버그
with st.sidebar:
    st.header("Configuration")
    st.write("**OPENAI_API_KEY set?**", bool(OPENAI_API_KEY))
    st.write("**OPENAI_PROJECT**:", OPENAI_PROJECT or "(empty)")
    st.write("**WORKFLOW/AGENT ID**:", WORKFLOW_OR_AGENT_ID or "(empty)")
    st.write("**BASE_MODEL**:", BASE_MODEL)

    st.markdown(
        """
        **필수 조건**
        1) `OPENAI_API_KEY`는 워크플로/에이전트가 속한 **같은 프로젝트**에서 발급  
        2) `OPENAI_PROJECT=proj_...`는 동일 프로젝트  
        3) `OPENAI_AGENT_ID`는 `wf_...` 또는 `agt_...` 그대로
        """
    )

    if st.button("Reset conversation", type="secondary"):
        st.session_state.history = []
        st.session_state.local_session_id = str(uuid.uuid4())
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

# 가드: 인증/ID 체크
if not OPENAI_API_KEY or not WORKFLOW_OR_AGENT_ID:
    st.info("환경변수 `OPENAI_API_KEY`, `OPENAI_PROJECT`, `OPENAI_AGENT_ID(wf_/agt_)`를 설정해 주세요.")
    st.stop()

# -------------------- Agent / Runner --------------------
# ⚠️ TypeError 방지를 위해 최소 옵션만 사용 (model_settings 제거)
base_agent = Agent(
    name="My agent",
    instructions="You are a helpful assistant.",
    model=BASE_MODEL,
)

def run_agent(user_text: str) -> str:
    """
    Agents SDK Runner로 워크플로 실행.
    - 워크플로/에이전트 호출은 Runner로 수행
    - workflow_id는 trace_metadata로 전달 (Agent Builder 'Get code' 흐름)
    """
    # 대화 입력(이번 턴만; 필요 시 과거 턴 누적도 가능)
    input_items = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }
    ]

    # Runner 생성 후 실행
    runner = Runner(
        trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": WORKFLOW_OR_AGENT_ID,  # ★ 핵심: wf_/agt_ 식별자 전달
            # 참고: 필요하면 여기 추가 메타데이터 넣을 수 있음
        }
    )

    result = runner.run(
        base_agent,
        input_items,
    )

    # SDK 버전에 따라 final_output 또는 finalOutput가 있을 수 있어 모두 확인
    final = getattr(result, "final_output", None) or getattr(result, "finalOutput", None) or ""
    return final.strip()

# -------------------- Chat UI --------------------
st.subheader("Chat")
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("메시지를 입력하세요…")

if prompt:
    # 사용자 메시지 렌더/저장
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
