def run_agent(user_text: str) -> str:
    """
    Agents SDK Runner로 실행.
    - workflow_id는 run_config.trace_metadata로 전달
    - final_output이 비어 있으면 new_items/raw_responses에서 폴백 파싱
    """
    input_items = [
        {"role": "user", "content": [{"type": "input_text", "text": user_text}]}
    ]

    result = runner.run(
        base_agent,
        input_items,
        run_config={
            "trace_metadata": {
                "__trace_source__": "agent-builder",
                "workflow_id": WORKFLOW_OR_AGENT_ID,  # wf_/agt_
            }
        },
    )

    # 1) 기본 경로: final_output / finalOutput
    text = getattr(result, "final_output", None) or getattr(result, "finalOutput", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 2) 폴백: new_items 안의 assistant 메시지에서 output_text 수집
    try:
        items = getattr(result, "new_items", []) or []
        chunks = []
        for it in items:
            raw = getattr(it, "raw_item", None) or {}
            if raw.get("role") != "assistant":
                continue
            for c in raw.get("content", []) or []:
                # Agents SDK는 content.type이 "output_text"로 들어옴
                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
        if chunks:
            return "\n".join(chunks).strip()
    except Exception:
        pass

    # 3) 마지막 폴백: raw_responses에서 텍스트 긁기 (SDK 버전별 임시 방어)
    try:
        raws = getattr(result, "raw_responses", []) or []
        for r in raws[::-1]:
            txt = r.get("text") or r.get("output_text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    except Exception:
        pass

    return "(no output_text)"
