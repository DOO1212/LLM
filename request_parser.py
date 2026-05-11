# request_parser.py

import json
import ollama

from prompts.sql_parser_prompt import (
    build_prompt
)

from prompts.entity_prompt import (
    KNOWN_ENTITIES,
    build_entity_injection_prompt
)

from config.llm_config import (
    LLM_MODEL,
    LLM_OPTIONS,
    AST_CACHE
)

from utils.ast_cache import (
    get_cached_ast,
    cache_ast
)


# ---------------- Entity Extraction ----------------

def extract_entities(query):

    entities = []


    for entity in KNOWN_ENTITIES:

        if entity in query:

            entities.append(entity)


    return entities


# ---------------- LLM 호출 ----------------

def ask_llm(query):

    # ---------------- Base Prompt ----------------

    prompt = build_prompt(query)


    # ---------------- Entity Extraction ----------------

    entities = extract_entities(query)


    # ---------------- Entity Prompt ----------------

    entity_prompt = build_entity_injection_prompt(
        entities
    )


    # ---------------- Prompt Merge ----------------

    if entity_prompt:

        prompt += "\n\n" + entity_prompt


    # ---------------- User Query ----------------

    prompt += (

        "\n\n"

        "질문:\n"

        + query
    )


    # ---------------- Prompt Debug ----------------

    print("\n[PROMPT]")
    print(prompt)


    # ---------------- LLM Request ----------------

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


    # ---------------- Response Content ----------------

    content = response["message"]["content"]


    # ---------------- Raw Output Debug ----------------

    print("\n[RAW LLM OUTPUT]")
    print(content)


    # ---------------- JSON Parse ----------------

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
        )
    }

    return ast


# ---------------- Query Parsing ----------------

def parse_query(query):


    # ---------------- AST Cache ----------------

    if AST_CACHE:

        cached_ast = get_cached_ast(query)

        if cached_ast:

            print("\n[AST CACHE HIT]")

            return cached_ast


    # ---------------- LLM Parsing ----------------

    llm_result = ask_llm(query)

    ast = build_ast(llm_result)


    # ---------------- AST Debug ----------------

    print("\n[AST]")
    print(ast)


    # ---------------- Cache Save ----------------

    if AST_CACHE:

        cache_ast(

            query,
            ast
        )


    return ast