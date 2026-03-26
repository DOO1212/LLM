import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

# Ollama(로컬 Llama 3.1) 설정
OLLAMA_MODEL = "llama3.1:8b"

def _question_op(question: str):
    # 자연어 조건을 수식 연산자로 변환
    if "이하" in question: return "<="
    if "미만" in question: return "<"
    if "이상" in question: return ">="
    if "초과" in question: return ">"
    return None

def normalize_panel(panel: dict, question: str):
    # LLM이 파싱한 JSON 결과를 시스템 표준(컬럼명, 연산자)에 맞게 강제 보정
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