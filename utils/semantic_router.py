# utils/semantic_router.py

from config.semantic_keywords import (

    SORT_KEYWORDS,

    FILTER_KEYWORDS,

    AGGREGATION_KEYWORDS,

    LIMIT_KEYWORDS
)


# ---------------- Keyword Matcher ----------------

def contains_keywords(

    query,
    keywords
):

    return any(

        keyword in query

        for keyword in keywords
    )


# ---------------- Semantic Detection ----------------

def detect_semantics(query):

    semantics = {

        "sorting": contains_keywords(

            query,
            SORT_KEYWORDS
        ),

        "filtering": contains_keywords(

            query,
            FILTER_KEYWORDS
        ),

        "aggregation": contains_keywords(

            query,
            AGGREGATION_KEYWORDS
        ),

        "limit": contains_keywords(

            query,
            LIMIT_KEYWORDS
        )
    }

    return semantics