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
  "table": "inventory",
  "aggregation": {
    "type": "SUM | AVG | COUNT | MAX | MIN",
    "column": "컬럼명"
  },
  "filters": [
    {
      "column": "컬럼명",
      "op": "=",
      "value": "값"
    }
  ],
  "sort": {
    "column": "컬럼명",
    "direction": "ASC | DESC"
  },
  "limit": 숫자,
  "unsupported": false
}

"""
