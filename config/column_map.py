from util.flatten import flatten_map

COLUMN_MAP = {
    "제품명": [
        "제품명", "제품 이름", "상품명", "상품 이름"
    ],
    "가격": [
        "가격", "금액"
    ],
    "재고수량": [
        "재고수량", "재고 수량"
    ],
    "입고일정": [
        "입고일정", "입고일", "입고 일정", "입고날짜", "입고 날짜"
    ],
    "입고수량": [
        "입고수량", "입고 수량"
    ]

}

ALL_TARGET_KEYWORDS = flatten_map(COLUMN_MAP)