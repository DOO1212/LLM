# request_parser.py

import json
import ollama

from prompts.god_prompt import GOD_PROMPT

from config.llm_config import (
    LLM_MODEL,
    LLM_OPTIONS
)

from config.entity_grounding import (
    ENTITY_GROUNDING
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

    return json.loads(content)


# ---------------- AST 생성 ----------------

def build_ast(llm_result):

    ast = {

        "table": "inventory",

        "aggregation": llm_result.get(
            "aggregation"
        ),

        "filters": llm_result.get(
            "filters",
            []
        ),

        "sort": llm_result.get(
            "sort"
        ),

        "limit": llm_result.get(
            "limit"
        ),

        "presentation_order": llm_result.get(
            "presentation_order"
        )
    }

    return ast


# ---------------- Entity Grounding ----------------

def apply_entity_grounding(

    query,
    ast
):

    filters = ast.get(
        "filters",
        []
    )

    # 이미 존재하는 filter 값
    existing_values = [

        str(
            f.get("value")
        )

        for f in filters
    ]


    for term, rule in ENTITY_GROUNDING.items():

        # query에 term 없으면 skip
        if term not in query:

            continue


        # 이미 존재하는 filter면 skip
        already_exists = any(

            term in value

            for value in existing_values
        )

        if already_exists:

            continue


        column = rule["column"]

        op = rule["op"]


        # LIKE
        if op == "LIKE":

            value = f"%{term}%"

        # =
        else:

            value = term


        filters.append({

            "column": column,

            "op": op,

            "value": value
        })


    ast["filters"] = filters

    return ast


# ---------------- Query Parsing ----------------

def parse_query(query):

    # ---------------- AST Cache ----------------

    cached_ast = get_cached_ast(query)

    if cached_ast:

        print("\n[AST CACHE HIT]")

        return cached_ast


    # ---------------- LLM Parsing ----------------

    llm_result = ask_llm(query)

    ast = build_ast(llm_result)


    # ---------------- Entity Grounding ----------------

    ast = apply_entity_grounding(

        query,
        ast
    )


    # ---------------- AST 출력 ----------------

    print("\n[AST]")
    print(ast)


    # ---------------- Cache 저장 ----------------

    cache_ast(

        query,
        ast
    )

    return ast