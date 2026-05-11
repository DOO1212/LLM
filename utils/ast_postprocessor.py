from prompts.semantic_keywords import (
    FILTER_KEYWORDS
)


def enrich_filters(ast, query):

    filters = ast.get("filters", [])


    # 이미 filter 있으면 그대로 사용
    if filters:

        return ast


    for keyword in FILTER_KEYWORDS:

        if keyword in query:

            filters.append({

                "column": "품목명",

                "op": "LIKE",

                "value": f"%{keyword}%"
            })

            break


    ast["filters"] = filters

    return ast