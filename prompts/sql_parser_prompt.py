# prompts/sql_parser_prompt.py

from prompts.system_prompt import SYSTEM_PROMPT

from prompts.schema_prompt import SCHEMA_PROMPT

from prompts.rule_prompt import RULE_PROMPT


# ---------------- Few-shot ----------------

from prompts.few_shot.sorting_example import (
    SORTING_EXAMPLES
)

from prompts.few_shot.filtering_example import (
    FILTERING_EXAMPLES
)

from prompts.few_shot.aggregation_example import (
    AGGREGATION_EXAMPLES
)

from prompts.few_shot.limit_example import (
    LIMIT_EXAMPLES
)


# ---------------- Semantic Router ----------------

from utils.semantic_router import (
    detect_semantics
)


# ---------------- Prompt Builder ----------------

def build_prompt(query):

    prompt = (

        SYSTEM_PROMPT

        + SCHEMA_PROMPT

        + RULE_PROMPT
    )


    # ---------------- Semantic Detection ----------------

    semantics = detect_semantics(query)


    # ---------------- Debug ----------------

    print("\n[SEMANTICS]")
    print(semantics)

    selected_modules = []


    # ---------------- Sorting ----------------

    if semantics["sorting"]:

        prompt += SORTING_EXAMPLES

        selected_modules.append(
            "SORTING"
        )


    # ---------------- Filtering ----------------

    if semantics["filtering"]:

        prompt += FILTERING_EXAMPLES

        selected_modules.append(
            "FILTERING"
        )


    # ---------------- Aggregation ----------------

    if semantics["aggregation"]:

        prompt += AGGREGATION_EXAMPLES

        selected_modules.append(
            "AGGREGATION"
        )


    # ---------------- Limit ----------------

    if semantics["limit"]:

        prompt += LIMIT_EXAMPLES

        selected_modules.append(
            "LIMIT"
        )


    # ---------------- Prompt Debug ----------------

    print("\n[PROMPT MODULES]")
    print(selected_modules)


    return prompt