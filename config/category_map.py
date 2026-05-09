from util.flatten import flatten_map

CATEGORY_MAP = {
    "전자": [
        "전자", "전자제품", "전자기기"
    ],
    "식품": [
        "식품", "음식", "먹거리"
    ],
    "의류": [
        "의류", "옷", "패션"
    ],
    "생활용품": [
        "생활", "생활용품"
    ],
    "사무용품": [
        "사무", "사무용품"
    ],
    "부품": [
        "부품", "파츠"
    ]
}

ALL_CATEGORY_KEYWORDS = flatten_map(CATEGORY_MAP)