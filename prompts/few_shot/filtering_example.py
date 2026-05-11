FILTERING_EXAMPLES = """

질문:
충전기 상품은?

출력:
{
    "aggregation": null,

    "filters": [
        {
            "column": "품목명",
            "op": "LIKE",
            "value": "%충전기%"
        }
    ],

    "sort": null,

    "limit": null,

    "presentation_order": null
}


질문:
전자 제품은?

출력:
{
    "aggregation": null,

    "filters": [
        {
            "column": "카테고리",
            "op": "=",
            "value": "전자"
        }
    ],

    "sort": null,

    "limit": null,

    "presentation_order": null
}

"""