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

"""