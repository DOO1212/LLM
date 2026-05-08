LIMIT_EXAMPLES = """

질문:
최근 입고된 상품 5개

출력:
{
    "aggregation": null,

    "filters": [],

    "sort": {
        "column": "입고일",
        "direction": "DESC"
    },

    "limit": 5,

    "presentation_order": null
}

"""