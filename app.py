"""
Gemini API 연동 채팅 (멀티턴).
실행: LLM 폴더에서  .venv/bin/python -m streamlit run app.py  →  http://localhost:8000
      (Ubuntu 시스템 Python은 pip 전역 설치 불가 → bash setup_venv.sh 로 .venv 생성)
키: LLM/.env 에 GEMINI_API_KEY=
"""

from pathlib import Path
from typing import Dict, List, Optional

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _get_api_key() -> Optional[str]:
    import os

    return os.environ.get("GEMINI_API_KEY")


def _to_gemini_history(messages: List[Dict[str, str]]) -> List[dict]:
    """messages[:-1] 구간을 Gemini start_chat용 history로 변환 (user/assistant 교대)."""
    h: List[dict] = []
    rest = messages[:-1]
    for i in range(0, len(rest), 2):
        if i + 1 >= len(rest):
            break
        u, m = rest[i], rest[i + 1]
        if u.get("role") != "user" or m.get("role") != "assistant":
            continue
        h.append({"role": "user", "parts": [u["content"]]})
        h.append({"role": "model", "parts": [m["content"]]})
    return h


st.set_page_config(page_title="Gemini 챗봇", page_icon="💬", layout="centered")
st.title("💬 Gemini 챗봇")
st.caption("Google Gemini와 대화합니다. API 키는 서버(로컬) 환경 변수로만 사용됩니다.")

api_key = _get_api_key()
if not api_key:
    st.error(
        "GEMINI_API_KEY가 없습니다.\n\n"
        "`LLM/.env`에 다음 한 줄을 넣으세요:\n\n"
        "`GEMINI_API_KEY=발급한_키`"
    )
    st.stop()

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []

model_name = st.sidebar.text_input("모델", value="gemini-2.5-pro", help="예: gemini-2.5-pro, gemini-2.0-flash")
if st.sidebar.button("대화 초기화"):
    st.session_state.gemini_messages = []
    st.rerun()

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)

for msg in st.session_state.gemini_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("메시지를 입력하세요")
if prompt:
    st.session_state.gemini_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            hist = _to_gemini_history(st.session_state.gemini_messages)
            chat = model.start_chat(history=hist)
            response = chat.send_message(st.session_state.gemini_messages[-1]["content"])
            text = (response.text or "").strip()
            if not text:
                text = "(빈 응답)"
            placeholder.markdown(text)
            st.session_state.gemini_messages.append({"role": "assistant", "content": text})
        except Exception as e:
            err = f"오류: {e}"
            placeholder.error(err)
            st.session_state.gemini_messages.append({"role": "assistant", "content": err})
