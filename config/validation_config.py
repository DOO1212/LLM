# validation_config.py


# ---------------- Schema ----------------

VALID_COLUMNS = [

    "재고ID",
    "상품코드",
    "품목명",
    "카테고리",
    "창고",
    "재고수량",
    "안전재고",
    "단가",
    "재고금액",
    "입고일",
    "최근출고일",
    "공급업체",
    "상태"
]


# ---------------- Filter Operators ----------------

VALID_FILTER_OPERATORS = [

    "=",
    "LIKE",
    ">",
    "<",
    ">=",
    "<="
]


# ---------------- Sort Directions ----------------

VALID_SORT_DIRECTIONS = [

    "ASC",
    "DESC"
]


# ---------------- Aggregation Types ----------------

VALID_AGGREGATIONS = [

    "sum",
    "average",
    "count",
    "max",
    "min"
]
