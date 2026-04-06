import json
import re
import requests
import numpy as np

from config.intent_map import ALL_INTENT_KEYWORDS
from config.target_map import ALL_TARGET_KEYWORDS
from config.column_map import ALL_COLUMN_KEYWORDS


# ---------------- 임베딩 모델 ----------------
model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# ---------------- 임베딩 캐싱 ----------------
def build_embedding_index(keyword_pairs):
    return [
        (key, kw, get_model().encode(kw))
        for key, kw in keyword_pairs
    ]


INTENT_EMBEDDINGS = build_embedding_index(ALL_INTENT_KEYWORDS)
TARGET_EMBEDDINGS = build_embedding_index(ALL_TARGET_KEYWORDS)
COLUMN_EMBEDDINGS = build_embedding_index(ALL_COLUMN_KEYWORDS)


# ---------------- 유사도 ----------------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------------- 공통 detect ----------------
def detect_from_keywords(query, keyword_pairs, embedding_index=None, use_embedding=False):

  
    for key, kw in keyword_pairs:
        if kw in query:
            return key

    # embedding fallback
    if use_embedding and embedding_index:
        query_vec = get_model().encode(query)

        best_key = None
        best_score = -1

        for key, kw, kw_vec in embedding_index:
            score = cosine_sim(query_vec, kw_vec)

            if score > best_score:
                best_score = score
                best_key = key

        # threshold
        if best_score > 0.5:
            return best_key

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

    # 코드 블록 제거
    output = re.sub(r"```json", "", output)
    output = re.sub(r"```", "", output).strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {
            "error": "llm_parse_failed",
            "raw_output": output
        }


# ---------------- 메인 ----------------
def parse_query(query):

    query = query.replace("상품들", "상품")

    intent = detect_from_keywords(query, ALL_INTENT_KEYWORDS, INTENT_EMBEDDINGS)
    target = detect_from_keywords(query, ALL_TARGET_KEYWORDS, TARGET_EMBEDDINGS, use_embedding=True)
    column = detect_from_keywords(query, ALL_COLUMN_KEYWORDS, COLUMN_EMBEDDINGS)

    filters = extract_filters(query)
    top_k = extract_top_k(query)
    return_column = extract_return_column(query)

    print(f"[DEBUG] intent={intent}, target={target}, column={column}, filters={filters}, top_k={top_k}, return_col={return_column}")

    if target == "products" and column is None:
        column = "가격"

    if intent and target and column:
        return {
            "intent": intent,
            "target": target,
            "column": column,
            "filters": filters,
            "top_k": top_k,
            "return_column": return_column
        }

    print("⚠️ LLM fallback 사용")
    return llm_parse(query)

def extract_filters(query):

    filters = []

    # 이상
    match = re.search(r"(\d+)\s*이상", query)
    if match:
        filters.append({
            "column": "가격",
            "op": ">=",
            "value": int(match.group(1))
        })

    # 이하
    match = re.search(r"(\d+)\s*이하", query)
    if match:
        filters.append({
            "column": "가격",
            "op": "<=",
            "value": int(match.group(1))
        })

    # 초과
    match = re.search(r"(\d+)\s*초과", query)
    if match:
        filters.append({
            "column": "가격",
            "op": ">",
            "value": int(match.group(1))
        })

    # 미만
    match = re.search(r"(\d+)\s*미만", query)
    if match:
        filters.append({
            "column": "가격",
            "op": "<",
            "value": int(match.group(1))
        })

    return filters


def apply_filters(df, filters):

    for f in filters:
        col = f["column"]
        op = f["op"]
        val = f["value"]

        if op == ">=":
            df = df[df[col] >= val]
        elif op == "<=":
            df = df[df[col] <= val]
        elif op == ">":
            df = df[df[col] > val]
        elif op == "<":
            df = df[df[col] < val]

    return df


def extract_top_k(query):
    match = re.search(r"(\d+)\s*개", query)
    if match:
        return int(match.group(1))
    return None

def extract_return_column(query):

    if "이름" in query or "상품명" in query:
        return "상품명"

    return None
