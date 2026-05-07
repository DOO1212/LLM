# request_parser.py

import ollama
import json
import re

from prompts.sql_parser_prompt import SQL_PARSER_PROMPT

from config.llm_config import (
    LLM_MODEL,
    LLM_OPTIONS
)


# ---------------- LLM 호출 ----------------

def ask_llm(query):

    prompt = f"""
    {SQL_PARSER_PROMPT}

    질문:
    {query}
    """

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

    # ---------------- markdown 제거 ----------------

    content = content.strip()

    content = content.replace(
        "```json",
        ""
    )

    content = content.replace(
        "```",
        ""
    )

    # ---------------- JSON 추출 ----------------

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


    operation = llm_result.get(
        "operation"
    )

    column = llm_result.get(
        "column"
    )

    filters = llm_result.get(
        "filters",
        []
    )

    limit = llm_result.get(
        "limit"
    )


    presentation_order = llm_result.get(
        "presentation_order"
    )

    aggregation = llm_result.get(
        "aggregation"
    )

    # ---------------- max/min 기본 limit ----------------

    if operation in ["max", "min"]:

        if not limit:

            limit = 1

    # ---------------- AST ----------------

    ast = {

        "table": "inventory",

        "aggregation" : aggregation,

        "operation": operation,

        "column": column,

        "filters": filters,

        "limit": limit,

        "presentation_order": presentation_order
    }

    return ast


# ---------------- 메인 Parser ----------------

def parse_query(query):

    # 1. LLM semantic parsing
    llm_result = ask_llm(query)

    # 2. AST 생성
    ast = build_ast(llm_result)

    print("\n[AST]")
    print(ast)

    return ast
