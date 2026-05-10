# prompts/few_shot/aggregation_example.py

AGGREGATION_EXAMPLES = """

질문:
재고수량 평균은?

출력:
{
    "aggregation": {
        "type": "average",
        "column": "재고수량"
    },

    "filters": [],

    "sort": null,

    "limit": null,

    "presentation_order": null
}


질문:
재고금액 총합은?

출력:
{
    "aggregation": {
        "type": "sum",
        "column": "재고금액"
    },

    "filters": [],

    "sort": null,

    "limit": null,

    "presentation_order": null
}


질문:
상품 개수는?

출력:
{
    "aggregation": {
        "type": "count",
        "column": "*"
    },

    "filters": [],

    "sort": null,

    "limit": null,

    "presentation_order": null
}

"""