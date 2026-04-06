import requests
import re

from config.intent_map import INTENT_MAP
from config.target_map import TARGET_MAP
from config.column_map import COLUMN_MAP

# ---------------- intent ----------------
def detect_intent(query):
    for key in INTENT_MAP:
        if key in query:
            return INTENT_MAP[key]
    return None

# ---------------- target ----------------
def detect_target(query):
    for key in TARGET_MAP:
        if key in query:
            return TARGET_MAP[key]
    return None

# ---------------- column ----------------
def detect_column(query):
    for key in COLUMN_MAP:
        if key in query:
            return COLUMN_MAP[key]
    return None

# ---------------- LLM fallback ----------------
def llm_parse(query):
    prompt = f"""
다음 질문에서 intent, target, column을 JSON으로 추출해라.

질문: {query}

형식:
{{
  "intent": "",
  "target": "",
  "column": ""
}}
"""
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        }
    )

    output = res.json().get("response", "").strip()

    # ```json 제거
    output = re.sub(r"```json", "", output)
    output = re.sub(r"```", "", output)

    return output


# ---------------- 메인 ----------------
def parse_query(query):

    intent = detect_intent(query)
    target = detect_target(query)
    column = detect_column(query)

    print(f"[DEBUG] intent={intent}, target={target}, column={column}")

    if not intent or not target or not column:
        print("⚠️ LLM fallback 사용")
        return llm_parse(query)

    return {
        "intent": intent,
        "target": target,
        "column": column
    }
