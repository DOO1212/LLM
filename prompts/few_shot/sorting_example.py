SORTING_EXAMPLES = """

질문:
가장 비싼 상품은?

출력:
{
    "aggregation": null,

    "filters": [],

    "sort": {
        "column": "단가",
        "direction": "DESC"
    },

    "limit": 1,

    "presentation_order": null
}


질문:
최근 입고된 상품 3개

출력:
{
    "aggregation": null,

    "filters": [],

    "sort": {
        "column": "입고일",
        "direction": "DESC"
    },

    "limit": 3,

    "presentation_order": null
}

"""