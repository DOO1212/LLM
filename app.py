"""
Gemini API 연동 채팅 및 Ollama 의도 해석 패널 통합 앱.
실행: .venv/bin/python -m streamlit run app.py
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import google.generativeai as genai
import requests
import streamlit as st
from dotenv import load_dotenv

# ==========================================
# 환경 설정 및 상수
# ==========================================
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Ollama 설정
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
ALLOWED_INTENTS = {"filter", "group_agg", "topk", "sort", "count", "unknown"}
ALLOWED_OPS = {"<", "<=", ">", ">=", "==", "contains"}
ALLOWED_COLUMNS = {
    "품목코드", "품목명", "카테고리", "현재수량", "안전재고", "단가(원)", "입고예정일", "상태",
}
COLUMN_ALIAS = {
    "가격": "단가(원)", "단가": "단가(원)", "price": "단가(원)", "재고": "현재수량", "수량": "현재수량",
}

st.set_page_config(page_title="통합 LLM 패널", page_icon="🚀", layout="centered")

# ==========================================
# 공통 상태 초기화
# ==========================================
if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []

# ==========================================
# 함수 정의 (Gemini용)
# ==========================================
def _get_api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY")

def _to_gemini_history(messages: List[Dict[str, str]]) -> List[dict]:
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

# ==========================================
# 함수 정의 (Ollama용)
# ==========================================
def _question_money(question: str):
    text = question.replace(",", "").replace(" ", "")
    values = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)만(\d+(?:\.\d+)?)천원", text):
        won = int(float(m.group(1)) * 10000 + float(m.group(2)) * 1000)
        values.append((m.start(), won))
    for m in re.finditer(r"(\d+(?:\.\d+)?)만원", text):
        won = int(float(m.group(1)) * 10000)
        values.append((m.start(), won))
    for m in re.finditer(r"(\d+(?:\.\d+)?)원", text):
        won = int(float(m.group(1)))
        values.append((m.start(), won))
    if not values: return None
    values.sort(key=lambda x: x[0])
    return values[-1][1]

def _question_op(question: str):
    if "이하" in question: return "<="
    if "미만" in question: return "<"
    if "이상" in question: return ">="
    if "초과" in question: return ">"
    return None

def _is_overview_question(question: str):
    keywords = ["현황", "요약", "전체", "총괄", "상태", "개요"]
    return any(k in question for k in keywords)

def normalize_panel(panel: dict, question: str):
    out = dict(panel) if isinstance(panel, dict) else {}
    intent = out.get("intent", "unknown")
    if intent not in ALLOWED_INTENTS: intent = "unknown"
    out["intent"] = intent

    conditions = out.get("conditions", [])
    if not isinstance(conditions, list): conditions = []

    if not conditions and isinstance(out, dict) and "column" in out and "op" in out:
        conditions = [{
            "column": out.get("column"), "op": out.get("op"),
            "value_won": out.get("value_won"), "value": out.get("value"),
        }]

    fixed = []
    for c in conditions:
        if not isinstance(c, dict): continue
        col = COLUMN_ALIAS.get(c.get("column"), c.get("column"))
        op = c.get("op")
        if op == "include": op = "contains"
        if op not in ALLOWED_OPS: continue
        if col not in ALLOWED_COLUMNS:
            if col in {"price", "가격", "단가"}: col = "단가(원)"
            else: continue

        if op == "contains":
            value = c.get("value")
            if not isinstance(value, str) or not value.strip(): continue
            fixed.append({"column": col, "op": op, "value": value.strip()})
        else:
            value_won = c.get("value_won")
            if value_won is None: continue
            try: value_won = int(value_won)
            except Exception: continue
            fixed.append({"column": col, "op": op, "value_won": value_won})

    q_money = _question_money(question)
    q_op = _question_op(question)
    forced_price_condition = False
    if q_money is not None and q_op is not None:
        fixed = [c for c in fixed if c.get("column") != "단가(원)"]
        fixed.append({"column": "단가(원)", "op": q_op, "value_won": q_money})
        forced_price_condition = True

    if _is_overview_question(question) and _question_money(question) is None and _question_op(question) is None:
        out["intent"] = "count"
        out["conditions"] = []
        out["reason_summary"] = "필터 조건 없이 전체 현황/요약 요청으로 해석"
        out["confidence"] = max(float(out.get("confidence", 0) or 0), 0.8)
        return out

    out["conditions"] = fixed
    out["confidence"] = float(out.get("confidence", 0) or 0)
    out["reason_summary"] = out.get("reason_summary", "")

    if forced_price_condition:
        op_ko = {"<=": "이하", "<": "미만", ">=": "이상", ">": "초과", "==": "동일"}.get(q_op, q_op)
        out["reason_summary"] = f"'단가(원) {q_money:,}원 {op_ko}' 조건으로 해석"

    return out

def parse_intent_panel(question: str):
    prompt = f"""너는 사용자 질문 의도 해석기다. 반드시 JSON 객체 1개만 출력하고, 다른 텍스트는 금지한다.
스키마:
{{
  "intent": "filter|group_agg|topk|sort|count|unknown",
  "conditions": [ {{"column":"단가(원)","op":"<|<=|>|>=|==|contains","value_won":50000,"value":"선택"}} ],
  "reason_summary": "짧은 근거 1~2문장", "confidence": 0.0
}}
규칙: 1만원=10000원, 0.5만원=5000원. 이하<=, 미만<, 이상>=, 초과>. confidence는 0~1. 질문: {question}"""
    
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_ctx": 1024, "num_predict": 256}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    raw = r.json().get("response", "").strip()
    m = re.search(r"\{[\s\S]*\}", raw.replace("```json", "").replace("```", "").strip())
    try: return json.loads(m.group(0)) if m else {"intent": "unknown", "conditions": []}
    except: return {"intent": "unknown", "conditions": [], "reason_summary": "파싱 실패"}

def repair_intent_panel(question: str, first_output: dict):
    prompt = f"""아래는 이전 파서 출력이다. 스키마 불일치/모호성을 수정해서 JSON 객체 1개만 다시 출력하라.
[질문] {question} \n[이전 출력] {json.dumps(first_output, ensure_ascii=False)}"""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    m = re.search(r"\{[\s\S]*\}", r.json().get("response", "").replace("```json", "").replace("```", "").strip())
    try: return json.loads(m.group(0)) if m else first_output
    except: return first_output

def verify_intent_panel(question: str, parsed_output: dict):
    prompt = f"""너는 의도 파싱 검증기다. 스키마/의미 일치 여부를 검증하라.
[질문] {question} \n[파싱 결과] {json.dumps(parsed_output, ensure_ascii=False)}
출력 스키마: {{"is_valid": true, "issues": ["문제 요약"], "corrected_panel": {{...}}}}"""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    m = re.search(r"\{[\s\S]*\}", r.json().get("response", "").replace("```json", "").replace("```", "").strip())
    try: return json.loads(m.group(0)) if m else {"is_valid": False, "issues": [], "corrected_panel": parsed_output}
    except: return {"is_valid": False, "issues": [], "corrected_panel": parsed_output}

# ==========================================
# UI 렌더링 (사이드바 메뉴)
# ==========================================
app_mode = st.sidebar.radio("메뉴 선택", ["💬 Gemini 챗봇", "🧠 의도 해석 패널 (Ollama)"])

if app_mode == "💬 Gemini 챗봇":
    st.title("💬 Gemini 챗봇")
    st.caption("Google Gemini와 대화합니다. API 키는 서버(로컬) 환경 변수로만 사용됩니다.")

    st.sidebar.divider()
    model_name = st.sidebar.text_input("Gemini 모델", value="gemini-2.5-pro", help="예: gemini-2.5-pro, gemini-2.0-flash")
    if st.sidebar.button("대화 초기화"):
        st.session_state.gemini_messages = []
        st.rerun()

    api_key = _get_api_key()
    if not api_key:
        st.error("GEMINI_API_KEY가 없습니다. `LLM/.env` 파일에 추가해 주세요.")
        st.stop()

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
                text = (response.text or "").strip() or "(빈 응답)"
                placeholder.markdown(text)
                st.session_state.gemini_messages.append({"role": "assistant", "content": text})
            except Exception as e:
                err = f"오류: {e}"
                placeholder.error(err)
                st.session_state.gemini_messages.append({"role": "assistant", "content": err})

elif app_mode == "🧠 의도 해석 패널 (Ollama)":
    st.title("🧠 의도 해석 패널")
    st.caption("답변 생성 없이, 질문 의도만 JSON으로 표시합니다. (Ollama 서버 필요)")

    q = st.text_input("질문 입력", "1만원은 10,000원이야. 5만원 이하 제품 알려줘")

    if st.button("의도 해석"):
        with st.spinner("해석 중..."):
            try:
                panel = normalize_panel(parse_intent_panel(q), q)
                intent = panel.get("intent", "unknown")
                conf = float(panel.get("confidence", 0) or 0)

                repaired = False
                if intent == "unknown" or conf < 0.6:
                    panel2 = normalize_panel(repair_intent_panel(q, panel), q)
                    intent2 = panel2.get("intent", "unknown")
                    conf2 = float(panel2.get("confidence", 0) or 0)
                    if (intent == "unknown" and intent2 != "unknown") or (conf2 > conf):
                        panel, intent, conf = panel2, intent2, conf2
                        repaired = True

                verified = verify_intent_panel(q, panel)
                verified_panel = normalize_panel(verified.get("corrected_panel", panel), q)
                if verified_panel != panel:
                    panel = verified_panel
                    intent = panel.get("intent", "unknown")
                    conf = float(panel.get("confidence", 0) or 0)

                st.success("의도 해석 완료")
                if repaired: st.info("자동 보정(self-repair) 1회 적용됨")
                if verified.get("is_valid", False): st.info("같은 모델 2차 검증 통과")
                else:
                    issues = verified.get("issues", [])
                    st.warning("2차 검증에서 보정됨" + (f": {'; '.join(str(x) for x in issues[:3])}" if issues else ""))

                st.json(panel)
                c1, c2 = st.columns(2)
                c1.metric("Intent", intent)
                c2.metric("Confidence", f"{conf:.2f}")
                st.write("**Reason**\n", panel.get("reason_summary", ""))

                if conf < 0.6:
                    st.warning("해석 신뢰도가 낮습니다. 질문을 더 구체화해 주세요.")

            except requests.exceptions.ConnectionError:
                st.error("Ollama 서버에 연결할 수 없습니다. (`ollama serve`가 실행 중인지 확인하세요)")
            except Exception as e:
                st.error(f"오류: {e}")