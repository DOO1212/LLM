# request_parser.py

import ollama
import json
import re

from prompts.sql_parser_prompt import build_prompt
from prompts.god_prompts import GOD_PROMPT

from config.llm_config import (
    LLM_MODEL,
    LLM_OPTIONS
)

from utils.ast_cache import (
    get_cached_ast,
    cache_ast
)


# ---------------- LLM 호출 ----------------

def ask_llm(query):

    prompt = GOD_PROMPT

    prompt += (
            "\n\n질문:\n"

            + query
     )

    response = ollama.chat(

        model=LLM_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        options=LLM_OPTIONS
    )

    content = response["message"]["content"]

    print("\n[RAW LLM OUTPUT]")
    print(content)

    return extract_json(content)


# ---------------- JSON 추출 ----------------

def extract_json(content):

    content = content.strip()

    # markdown 제거
    content = content.replace(
        "```json",
        ""
    )

    content = content.replace(
        "```",
        ""
    )

    # JSON 추출
    match = re.search(
        r"\{.*\}",
        content,
        re.DOTALL
    )

    if not match:

        raise ValueError(
            "JSON 추출 실패"
        )

    json_str = match.group()

    return json.loads(json_str)


# ---------------- AST 생성 ----------------

def build_ast(llm_result):

    aggregation = llm_result.get(
        "aggregation"
    )

    filters = llm_result.get(
        "filters",
        []
    )

    sort = llm_result.get(
        "sort"
    )

    limit = llm_result.get(
        "limit"
    )

    presentation_order = llm_result.get(
        "presentation_order"
    )

    # ---------------- AST ----------------

    ast = {

        "table": "inventory",

        "aggregation": aggregation,

        "filters": filters,

        "sort": sort,

        "limit": limit,

        "presentation_order": presentation_order
    }

    return ast


# ---------------- Parser ----------------

def parse_query(query):

    # ---------------- 캐시 조회 ----------------

    cached_ast = get_cached_ast(query)

    if cached_ast:

        print("\n[AST CACHE HIT]")

        return cached_ast


    # ---------------- LLM Parsing ----------------

    llm_result = ask_llm(query)

    ast = build_ast(llm_result)


    # ---------------- 캐시 저장 ----------------

    cache_ast(query, ast)


    print("\n[AST]")
    print(ast)

    return ast
