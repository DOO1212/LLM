from util.flatten import flatten_map

TARGET_MAP = {
    "inventory": [
        "상품", "상품들", "제품", "제품들", "물건", "물건들"
        ]
}

ALL_TARGET_KEYWORDS = flatten_map(TARGET_MAP)