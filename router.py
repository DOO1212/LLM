import pandas as pd
import os

from config.file_map import FILE_PATH_MAP
from request_parser import apply_filters

def route(parsed):

    intent = parsed.get("intent")
    target = parsed.get("target")
    column = parsed.get("column")
    filters = parsed.get("filters", [])
    top_k = parsed.get("top_k")
    return_column = parsed.get("return_column")

    file_path = FILE_PATH_MAP.get(target)

    if not file_path or not os.path.exists(file_path):
        return {"error": "file not found"}

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    if column not in df.columns:
        return {"error": f"column not found: {column}"}

    # 필터 적용
    if filters:
        df = apply_filters(df, filters)

    # Top N 처리
    if top_k and intent in ["max", "min"]:

        ascending = True if intent == "min" else False

        df_sorted = df.sort_values(by=column, ascending=ascending).head(top_k)

        if return_column and return_column in df.columns:
            return df_sorted[[return_column, column]].to_dict(orient="records")

        return df_sorted.to_dict(orient="records")

    # 일반 연산
    if intent == "average":
        result = df[column].mean()

    elif intent == "sum":
        result = df[column].sum()

    elif intent == "max":
        result = df[column].max()

    elif intent == "min":
        result = df[column].min()

    else:
        return {"error": "unknown intent"}

    return {
        "intent": intent,
        "target": target,
        "column": column,
        "filters": filters,
        "result": result
    }

