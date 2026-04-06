import pandas as pd
import os
from config.file_map import FILE_PATH_MAP

def route(parsed):

    # parsed가 string이면 변환
    if isinstance(parsed, str):
        import json
        parsed = json.loads(parsed)
        print(f"[ROUTER] {parsed}")

    intent = parsed.get("intent")
    target = parsed.get("target")
    column = parsed.get("column")

    # 예외 처리
    if not intent or not target or not column:
        return {"error": "missing fields", "parsed": parsed}

    # 파일 경로 찾기
    file_path = FILE_PATH_MAP.get(target)

    if not file_path or not os.path.exists(file_path):
        return {"error": f"file not found: {target}"}

    # 데이터 로드
    df = pd.read_excel(file_path)

    if column not in df.columns:
        print("📊 columns:", df.columns)
        return {"error": f"column not found: {column}"}

    # 핵심 연산
    if intent == "average":
        result = df[column].mean()

    elif intent == "sum":
        result = df[column].sum()

    elif intent == "max":
        result = df[column].max()

    elif intent == "min":
        result = df[column].min()

    else:
        return {"error": f"unknown intent: {intent}"}

    return {
            
        "intent": intent,
        "target": target,
        "column": column,
        "result": result
    }



