FILTERING_EXAMPLES = """

질문:
전자제품 중 가장 싼 충전기

출력:
{
    "aggregation": null,

    "filters": [
        {
            "column": "카테고리",
            "op": "=",
            "value": "전자"
        },
        {
            "column": "품목명",
            "op": "LIKE",
            "value": "%충전기%"
        }
    ],

    "sort": {
        "column": "단가",
        "direction": "ASC"
    },

    "limit": 1,

    "presentation_order": null
}

"""