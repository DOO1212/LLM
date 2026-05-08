# prompts/sql_parser_prompt.py

from prompts.system_prompt import SYSTEM_PROMPT

from prompts.schema_prompt import SCHEMA_PROMPT

from prompts.rule_prompt import RULE_PROMPT


# ---------------- Few-shot ----------------

from prompts.few_shot.sorting_examples import (
    SORTING_EXAMPLES
)

from prompts.few_shot.filtering_examples import (
    FILTERING_EXAMPLES
)

from prompts.few_shot.aggregation_examples import (
    AGGREGATION_EXAMPLES
)

from prompts.few_shot.limit_examples import (
    LIMIT_EXAMPLES
)


# ---------------- Keyword Config ----------------

from config.sorting_keywords import (
    SORTING_KEYWORDS
)

from config.filtering_keywords import (
    FILTERING_KEYWORDS
)

from config.aggregation_keywords import (
    AGGREGATION_KEYWORDS
)

from config.limit_keywords import (
    LIMIT_KEYWORDS
)


# ---------------- Prompt Builder ----------------

def build_prompt(query):

    prompt = (

        SYSTEM_PROMPT

        + SCHEMA_PROMPT

        + RULE_PROMPT
    )


    # ---------------- Sorting ----------------

    if any(

        keyword in query

        for keyword in SORTING_KEYWORDS
    ):

        prompt += SORTING_EXAMPLES


    # ---------------- Filtering ----------------

    if any(

        keyword in query

        for keyword in FILTERING_KEYWORDS
    ):

        prompt += FILTERING_EXAMPLES


    # ---------------- Aggregation ----------------

    if any(

        keyword in query

        for keyword in AGGREGATION_KEYWORDS
    ):

        prompt += AGGREGATION_EXAMPLES


    # ---------------- Limit ----------------

    if any(

        keyword in query

        for keyword in LIMIT_KEYWORDS
    ):

        prompt += LIMIT_EXAMPLES


    return prompt