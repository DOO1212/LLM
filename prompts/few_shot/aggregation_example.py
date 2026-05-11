# prompts/few_shot/aggregation_example.py

AGGREGATION_EXAMPLES = """

질문:
재고수량 평균은?

출력:
{
    "aggregation": {
        "type": "AVG",
        "column": "재고수량"
    },

    "filters": [],

    "sort": null,

    "limit": null,

    "unsupported": false
}


질문:
재고금액 총합은?

출력:
{
    "aggregation": {
        "type": "SUM",
        "column": "재고금액"
    },

    "filters": [],

    "sort": null,

    "limit": null,

    "unsupported": false
}


질문:
패딩 재고 수량 총합은?

출력:
{
    "aggregation": {
        "type": "SUM",
        "column": "재고수량"
    },

    "filters": [
        {
            "column": "품목명",
            "op": "LIKE",
            "value": "%패딩%"
        }
    ],

    "sort": null,

    "limit": null,

    "unsupported": false
}


질문:
현재 남아있는 충전기 개수는?

출력:
{
    "aggregation": {
        "type": "COUNT",
        "column": "재고수량"
    },

    "filters": [
        {
            "column": "품목명",
            "op": "LIKE",
            "value": "%충전기%"
        }
    ],

    "sort": null,

    "limit": null,

    "unsupported": false
}

"""