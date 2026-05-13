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

from column_selector import (
    ColumnSelector
)


# ---------------- Column Selector ----------------

selector = ColumnSelector()


# ---------------- Entity Extraction ----------------

def extract_entities(query):

    entities = []

    for entity in KNOWN_ENTITIES:

        if entity in query:

            entities.append(entity)

    return entities


# ---------------- Prompt Build ----------------

def build_final_prompt(

    query,
    selected_columns,
    entities
):

    # ---------------- Base Prompt ----------------

    prompt = build_prompt(query)


    # ---------------- Available Columns ----------------

    if selected_columns:

        prompt += (

            "\n\n"

            "[사용 가능한 컬럼]\n"

            + "\n".join(

                f"- {column}"

                for column in selected_columns
            )
        )


    # ---------------- Entity Prompt ----------------

    entity_prompt = build_entity_injection_prompt(
        entities
    )

    if entity_prompt:

        prompt += "\n\n" + entity_prompt


    # ---------------- User Query ----------------

    prompt += (

        "\n\n"

        "질문:\n"

        + query
    )

    return prompt


# ---------------- LLM Call ----------------

def ask_llm(

    query,
    selected_columns
):

    # ---------------- Entity Extraction ----------------

    entities = extract_entities(query)


    # ---------------- Prompt Build ----------------

    prompt = build_final_prompt(

        query=query,

        selected_columns=selected_columns,

        entities=entities
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


# ---------------- AST Build ----------------

def build_ast(

    llm_result,
    selected_info
):

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

        # ---------------- Selector Metadata ----------------

        "selected_columns": selected_info.get(
            "columns",
            []
        ),

        "return_column": selected_info.get(
            "return_column"
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


    # ---------------- Column Selection ----------------

    selected_info = selector.select_columns(
        query
    )


    # ---------------- Column Debug ----------------

    print("\n[COLUMN SELECTOR]")
    print(selected_info)


    selected_columns = selected_info.get(
        "columns",
        []
    )


    # ---------------- LLM Parsing ----------------

    llm_result = ask_llm(

        query=query,

        selected_columns=selected_columns
    )


    # ---------------- AST Build ----------------

    ast = build_ast(

        llm_result=llm_result,

        selected_info=selected_info
    )


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