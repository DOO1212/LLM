from util.flatten import flatten_map

INTENT_MAP = {
    "average": ["평균"],
    "sum": ["합", "합계", "총합", "총액", "합산", "총 합계", "총 합산"],
    "max": ["최대", "최고", "가장 비싼", "제일 비싼", "가장 높은", "제일 높은"],
    "min": ["최소", "최저", "가장 싼", "제일 싼", "가장 낮은", "제일 저렴한", "가장 저렴한"]
}

ALL_INTENT_KEYWORDS = flatten_map(INTENT_MAP)
