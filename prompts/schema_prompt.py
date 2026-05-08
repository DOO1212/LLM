# prompts/schema_prompt.py

SCHEMA_PROMPT = """

--------------------------------------------------
[DATABASE]
--------------------------------------------------

table:
- inventory

columns:
- 재고ID
- 상품코드
- 품목명
- 카테고리
- 창고
- 재고수량
- 안전재고
- 단가
- 재고금액
- 입고일
- 최근출고일
- 공급업체
- 상태

--------------------------------------------------
[OUTPUT SCHEMA]
--------------------------------------------------

{
    "aggregation": null,

    "filters": [],

    "sort": {
        "column": "...",
        "direction": "ASC | DESC"
    },

    "limit": null,

    "presentation_order": null
}

"""