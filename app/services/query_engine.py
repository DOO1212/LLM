import re


# 컬럼 의미를 canonical metric으로 정규화하기 위한 사전
COLUMN_ALIASES = {
    "item_name": ["품목", "제품", "자재", "코드", "명칭", "이름", "item", "name"],
    "min_order_qty": ["최소발주량", "최소주문", "최소수량", "moq"],
    "unit_price": ["단가", "가격", "원가", "unit_price", "price"],
    "amount": ["금액", "amount", "합계금액", "총금액"],
    "quantity": ["수량", "개수", "물량", "재고"],
}

# 질문 의도를 operation으로 정규화
OPERATION_PATTERNS = {
    "total_amount_min_order": [
        "최소발주량총금액",
        "최소주문총금액",
        "최소발주총금액",
        "최소발주량금액",
        "최소주문금액",
    ]
}

STOPWORDS = {
    "얼마", "얼마야", "얼마지", "알려줘", "보여줘", "확인", "조회", "이", "그", "저",
    "품목", "최소발주량", "최소주문", "최소수량", "총금액", "금액", "총액",
}

DOMAIN_KEYWORDS = {
    "재고": [
        "재고", "입고", "출고", "반품", "발주", "자재", "창고", "품목", "안전재고", "재주문", "로트",
    ],
    "생산": [
        "생산", "공정", "라인", "가동", "설비", "불량", "작업지시", "작업지", "실적", "수율", "계획", "downtime",
    ],
    "재무": [
        "재무", "매출", "매입", "비용", "예산", "손익", "원가", "정산", "전표", "미수금", "미지급",
    ],
    "규율": [
        "규정", "규율", "근태", "출근", "퇴근", "지각", "조퇴", "연차", "휴가", "보안", "복무",
    ],
    "기타": [
        "공지", "게시판", "포털", "사내", "부서", "조직도", "안내", "양식",
    ],
}


def _count_domain_hits(query: str, keywords: list[str]) -> int:
    q = (query or "").replace(" ", "").lower()
    return sum(1 for kw in keywords if kw.lower() in q)


def _to_number(value):
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _match_metric_columns(columns: list[str]) -> dict[str, int]:
    matched: dict[str, int] = {}
    lowered = [str(c).lower() for c in columns]
    for metric, aliases in COLUMN_ALIASES.items():
        for i, col in enumerate(lowered):
            if any(alias in col for alias in aliases):
                matched[metric] = i
                break
    return matched


def _detect_operation(query: str) -> str | None:
    q = (query or "").replace(" ", "").lower()
    for op, patterns in OPERATION_PATTERNS.items():
        if any(p in q for p in patterns):
            return op
    # 완전 고정 문구가 아니어도 최소발주량 + 금액 조합이면 같은 연산으로 해석
    if ("최소발주량" in q or "최소주문" in q or "moq" in q) and ("금액" in q or "총액" in q):
        return "total_amount_min_order"
    return None


def infer_label_hint(query: str) -> dict | None:
    """
    질문 해석 결과를 기반으로 라우터에 도메인 힌트를 제공합니다.
    예: 최소발주량 총금액 계산 질의 -> 재고 도메인
    """
    op = _detect_operation(query)
    if op == "total_amount_min_order":
        return {"label": "재고", "reason": "operation:total_amount_min_order", "boost": 0.18}

    # 일반 키워드 기반 힌트: 도메인 신호가 2개 이상일 때만 적용해 오탐을 줄입니다.
    hit_counts = {
        label: _count_domain_hits(query, keywords)
        for label, keywords in DOMAIN_KEYWORDS.items()
    }
    best_label, best_hits = max(hit_counts.items(), key=lambda x: x[1])
    if best_hits >= 2:
        boost = min(0.12 + 0.05 * (best_hits - 2), 0.28)
        return {
            "label": best_label,
            "reason": f"keyword_hits:{best_label}:{best_hits}",
            "boost": round(boost, 3),
        }
    return None


def _extract_entity_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9가-힣][\w가-힣x\-\.]*", query or "")
    cleaned = []
    for token in tokens:
        low = token.lower()
        if low in STOPWORDS:
            continue
        if len(low) < 2:
            continue
        cleaned.append(token)
    return cleaned


def _pick_target_row(rows: list[list], columns: list[str], metric_map: dict[str, int], query: str):
    if not rows:
        return None
    name_idx = metric_map.get("item_name")
    if name_idx is None:
        return rows[0]

    tokens = _extract_entity_tokens(query)
    if not tokens:
        return rows[0]

    best_row = None
    best_score = -1
    for row in rows:
        if name_idx >= len(row):
            continue
        text = str(row[name_idx]).lower()
        score = sum(1 for t in tokens if t.lower() in text)
        if score > best_score:
            best_score = score
            best_row = row
    return best_row if best_score > 0 else rows[0]


def compute_structured_answer(query: str, label: str, table_data: dict | None) -> str | None:
    """
    질문을 구조화(의도/지표/대상)해서 계산형 답변을 생성합니다.
    지원되지 않는 질문은 None 반환 -> 기존 분기/LLM이 처리.
    """
    if not table_data or not table_data.get("rows"):
        return None
    if label not in ("재고", "생산", "재무"):
        return None

    operation = _detect_operation(query)
    if not operation:
        return None

    columns = table_data.get("columns", [])
    rows = table_data.get("rows", [])
    metric_map = _match_metric_columns(columns)
    row = _pick_target_row(rows, columns, metric_map, query)
    if row is None:
        return None

    summary = table_data.get("summary", "")

    if operation == "total_amount_min_order":
        name_idx = metric_map.get("item_name")
        min_qty_idx = metric_map.get("min_order_qty")
        unit_price_idx = metric_map.get("unit_price")
        amount_idx = metric_map.get("amount")

        item_name = row[name_idx] if name_idx is not None and name_idx < len(row) else "선택 품목"
        min_qty = _to_number(row[min_qty_idx]) if min_qty_idx is not None and min_qty_idx < len(row) else None
        unit_price = _to_number(row[unit_price_idx]) if unit_price_idx is not None and unit_price_idx < len(row) else None
        amount = _to_number(row[amount_idx]) if amount_idx is not None and amount_idx < len(row) else None

        if amount is None and min_qty is not None and unit_price is not None:
            amount = min_qty * unit_price
        if amount is None:
            return None

        lines = [summary, f"품목 '{item_name}' 기준 최소발주량 총 금액은 {amount:,.0f}원입니다."]
        if min_qty is not None:
            lines.append(f"- 최소발주량: {min_qty:,.0f}")
        if unit_price is not None:
            lines.append(f"- 단가: {unit_price:,.0f}원")
        return "\n".join(lines)

    return None
