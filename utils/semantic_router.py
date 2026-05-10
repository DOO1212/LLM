from config.semantic_keywords import (

    SORT_KEYWORDS,

    FILTER_KEYWORDS,

    AGGREGATION_KEYWORDS
)


def contains_keywords(

    query,
    keywords
):

    return any(

        keyword in query

        for keyword in keywords
    )


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
        )
    }

    return semantics