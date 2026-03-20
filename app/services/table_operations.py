"""
표 조회 결과(table_data)에 대한 백엔드 연산 레지스트리.

- 숫자·집계·필터·특정 셀 조회는 여기서 처리하고, LLM은 서술만 담당하는 방향으로 확장한다.
- 연산 추가: OPERATION_SPECS에 한 줄 요약을 적고, _try_* 함수를 구현한 뒤
  OPERATION_RUNNERS 튜플에 등록한다.
"""
from __future__ import annotations

import re
from typing import Callable

# 문서·대시보드용 (API로 노출할 때 참고)
OPERATION_SPECS: list[dict[str, str]] = [
    {"id": "count_unit_price_equals", "desc": "단가(원)가 정확히 N원인 행 개수"},
    {"id": "count_category_value", "desc": "카테고리(또는 유사 열) 값이 특정 문자열인 행 개수"},
    {"id": "count_column_equals_number", "desc": "특정 열 값이 정수 N과 같은 행 개수 (예: 최소발주량 3)"},
    {"id": "count_unit_equals", "desc": "단위 열이 특정 값(예: kg)인 행 개수"},
    {"id": "count_date_column_equals", "desc": "날짜 열이 특정 일자인 행 개수 (예: 입고예정일)"},
    {"id": "count_manager_items", "desc": "담당자 이름이 일치하는 행 개수"},
    {"id": "min_unit_price_in_category", "desc": "카테고리 내 단가 최저 품목명"},
    {"id": "sum_column_by_date", "desc": "작업일자가 특정일인 행들의 불량수(또는 지정 열) 합계"},
    {"id": "max_column", "desc": "특정 숫자 열의 최댓값 (예: 다운타임분)"},
    {"id": "production_cell", "desc": "작업일자+라인+공정+지표명으로 단일 셀 조회"},
    {"id": "finance_metric", "desc": "기준월/부서/계정과목 조건으로 단일 금액·건수 조회"},
]


def _to_number(value) -> float | None:
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _col_index(columns: list[str], keywords: tuple[str, ...]) -> int | None:
    """키워드가 컬럼명에 포함되면 매칭. 긴 키워드를 먼저 검사(예: 품목명 vs 품목코드)."""
    for k in sorted(keywords, key=len, reverse=True):
        for i, c in enumerate(columns):
            if k in str(c):
                return i
    return None


def _col_item_display_name(columns: list[str]) -> int | None:
    """표시용 품목 이름 열 (품목코드 오매칭 방지)."""
    for i, c in enumerate(columns):
        cn = str(c).strip()
        if "품목명" in cn and "코드" not in cn:
            return i
    return _col_index(columns, ("제품명", "자재명", "명칭", "이름"))


def _find_date_in_query(query: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", query or "")
    return m.group(1) if m else None


def _find_month_in_query(query: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2})", query or "")
    return m.group(1) if m else None


def _try_count_unit_price_equals(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    q = (query or "").replace(" ", "")
    m = re.search(r"정확히\s*(\d+)\s*원", query or "")
    if not m:
        m = re.search(r"(\d{4,})원인", q)
    if not m:
        return None
    target = float(m.group(1))
    idx = _col_index(columns, ("단가", "가격", "원"))
    if idx is None:
        return None
    if "몇" not in q and "건" not in q and "개수" not in q:
        return None
    cnt = sum(
        1
        for row in rows
        if idx < len(row) and _to_number(row[idx]) == target
    )
    lines = [summary, f"단가가 {target:,.0f}원인 행은 {cnt}건입니다."]
    return "\n".join(lines)


def _try_count_category_value(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    m = re.search(r"카테고리.*?['\"]?([^'\"\\s]+)['\"]?.{0,6}몇", query or "")
    if not m:
        m = re.search(r"카테고리가\s*['\"]?([^'\"]+?)['\"]?\s*인", query or "")
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw.endswith("인"):
        cat = raw
    else:
        cat = raw[:-1].strip()
    idx = _col_index(columns, ("카테고리", "분류", "category"))
    if idx is None:
        return None
    q = (query or "").replace(" ", "")
    if "몇" not in q and "개수" not in q and "숫자" not in q:
        return None
    cnt = sum(1 for row in rows if idx < len(row) and cat in str(row[idx]))
    lines = [summary, f"카테고리가 '{cat}'인 품목은 {cnt}개입니다."]
    return "\n".join(lines)


def _try_count_column_equals_number(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    m = re.search(
        r"(최소발주량|현재수량|안전재고)\s*가?\s*(\d+)\s*인",
        query or "",
    )
    if not m:
        return None
    col_kw = m.group(1)
    target = int(m.group(2))
    idx = _col_index(columns, (col_kw,))
    if idx is None:
        return None
    q = (query or "").replace(" ", "")
    wants_code = "품목코드" in q or "코드" in q
    if wants_code:
        for row in rows:
            if idx < len(row) and _to_number(row[idx]) == float(target):
                code_idx = _col_index(columns, ("품목코드", "코드", "item"))
                if code_idx is not None and code_idx < len(row):
                    return "\n".join([summary, f"조건에 맞는 품목코드는 {row[code_idx]}입니다."])
        return "\n".join([summary, "조건에 맞는 품목코드가 없습니다."])
    if "몇" not in q and "건" not in q:
        return None
    cnt = sum(1 for row in rows if idx < len(row) and _to_number(row[idx]) == float(target))
    return "\n".join([summary, f"{col_kw}이(가) {target}인 행은 {cnt}건입니다."])


def _try_count_unit_equals(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    m = re.search(r"단위가\s*(\w+)\s*인", query or "")
    if not m:
        return None
    unit = m.group(1).strip()
    idx = _col_index(columns, ("단위", "unit"))
    if idx is None:
        return None
    q = (query or "").replace(" ", "")
    if "몇" not in q and "건" not in q:
        return None
    cnt = sum(1 for row in rows if idx < len(row) and str(row[idx]).strip() == unit)
    return "\n".join([summary, f"단위가 '{unit}'인 행은 {cnt}건입니다."])


def _try_count_date_column_equals(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    ds = _find_date_in_query(query or "")
    if not ds:
        return None
    q = (query or "").replace(" ", "")
    if "몇" not in q and "건" not in q:
        return None
    date_idx = _col_index(columns, ("입고예정일", "예정일", "기준일자", "작업일자", "일자"))
    if date_idx is None:
        return None
    cnt = 0
    for row in rows:
        if date_idx >= len(row):
            continue
        cell = str(row[date_idx]).strip()
        if cell.startswith(ds) or ds in cell:
            cnt += 1
    return "\n".join([summary, f"날짜 조건({ds})에 해당하는 행은 {cnt}건입니다."])


def _try_count_manager_items(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    m = re.search(r"담당자.*?([가-힣]{2,4})\s*인", query or "")
    if not m:
        m = re.search(r"담당자가\s*([가-힣]{2,4})", query or "")
    if not m:
        return None
    name = m.group(1).strip()
    if name.endswith("인"):
        name = name[:-1]
    idx = _col_index(columns, ("담당자", "관리", "책임"))
    if idx is None:
        return None
    q = (query or "").replace(" ", "")
    if "몇" not in q and "개수" not in q:
        return None
    cnt = sum(1 for row in rows if idx < len(row) and name in str(row[idx]))
    return "\n".join([summary, f"담당자가 '{name}'인 품목은 {cnt}개입니다."])


def _try_min_unit_price_in_category(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    if "단가" not in (query or "") or "낮" not in (query or "") and "최저" not in (query or ""):
        return None
    cat_idx = _col_index(columns, ("카테고리", "분류"))
    price_idx = _col_index(columns, ("단가", "가격"))
    name_idx = _col_item_display_name(columns)
    if cat_idx is None or price_idx is None or name_idx is None:
        return None
    q = query or ""
    target_cat = None
    for cat in ("유압/공압", "베어링", "원자재", "소모품", "전기/전자"):
        if cat in q:
            target_cat = cat
            break
    if not target_cat:
        return None
    best_price = None
    best_name = None
    for row in rows:
        if cat_idx >= len(row) or price_idx >= len(row) or name_idx >= len(row):
            continue
        if target_cat not in str(row[cat_idx]):
            continue
        p = _to_number(row[price_idx])
        if p is None:
            continue
        if best_price is None or p < best_price:
            best_price = p
            best_name = str(row[name_idx]).strip()
    if best_name is None:
        return None
    return "\n".join(
        [summary, f"{target_cat} 카테고리에서 단가가 가장 낮은 품목은 '{best_name}' ({best_price:,.0f}원)입니다."]
    )


def _try_sum_column_by_date(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    ds = _find_date_in_query(query or "")
    if not ds:
        return None
    q = (query or "").replace(" ", "")
    if "불량" not in q and "합" not in q:
        return None
    date_idx = _col_index(columns, ("작업일자", "일자", "날짜"))
    bad_idx = _col_index(columns, ("불량",))
    if date_idx is None or bad_idx is None:
        return None
    total = 0.0
    for row in rows:
        if date_idx >= len(row) or bad_idx >= len(row):
            continue
        cell = str(row[date_idx]).strip()
        if not (cell.startswith(ds) or ds in cell):
            continue
        v = _to_number(row[bad_idx])
        if v is not None:
            total += v
    return "\n".join([summary, f"{ds} 기준 불량수 합계는 {total:,.0f}입니다."])


def _try_max_column(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    q = query or ""
    if "다운타임" not in q and "최대" not in q and "가장" not in q:
        return None
    idx = _col_index(columns, ("다운타임", "downtime"))
    if idx is None:
        return None
    vals = []
    for row in rows:
        if idx < len(row):
            v = _to_number(row[idx])
            if v is not None:
                vals.append(v)
    if not vals:
        return None
    mx = max(vals)
    return "\n".join([summary, f"다운타임분 최댓값은 {mx:,.0f}분입니다."])


def _try_production_cell(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    ds = _find_date_in_query(query or "")
    if not ds:
        return None
    line = None
    for L in ("A라인", "B라인", "C라인"):
        if L in (query or ""):
            line = L
            break
    proc = None
    for p in ("조립", "검사", "가공", "포장"):
        if p in (query or ""):
            proc = p
            break
    metric_col = None
    qn = query or ""
    if "생산량" in qn:
        metric_col = "생산량"
    elif "불량수" in qn or ("불량" in qn and "합" not in qn.replace(" ", "")):
        metric_col = "불량수"
    elif "가동률" in qn:
        metric_col = "가동률"
    elif "목표달성률" in qn or "달성률" in qn:
        metric_col = "목표달성률"
    elif "다운타임" in qn:
        metric_col = "다운타임분"
    if not line or not proc or not metric_col:
        return None

    ci_line = _col_index(columns, ("라인",))
    ci_proc = _col_index(columns, ("공정",))
    ci_date = _col_index(columns, ("작업일자", "일자"))
    ci_metric = _col_index(columns, (metric_col,))
    if None in (ci_line, ci_proc, ci_date, ci_metric):
        return None

    for row in rows:
        if max(ci_line, ci_proc, ci_date, ci_metric) >= len(row):
            continue
        dcell = str(row[ci_date]).strip()
        if not (dcell.startswith(ds) or ds in dcell):
            continue
        if str(row[ci_line]).strip() != line:
            continue
        if proc not in str(row[ci_proc]):
            continue
        val = row[ci_metric]
        unit = "%" if "률" in metric_col else ("분" if "다운타임" in metric_col else "")
        suf = unit if unit else ""
        return "\n".join([summary, f"{ds} {line} {proc} 공정의 {metric_col}은(는) {val}{suf}입니다."])
    return "\n".join([summary, "조건에 맞는 행이 없습니다."])


def _try_finance_metric(query: str, columns: list[str], rows: list[list], summary: str) -> str | None:
    q = query or ""
    month = _find_month_in_query(q)
    if not month:
        return None
    dept = None
    for d in ("영업", "생산", "관리", "물류"):
        if d in q:
            dept = d
            break
    if not dept:
        return None

    want = None
    if "전표" in q or "건수" in q:
        want = "전표처리건수"
    elif "비용" in q:
        want = "비용"
    elif "매출" in q:
        want = "매출"
    elif "예산" in q:
        want = "예산"
    elif "미수금" in q:
        want = "미수금"
    elif "손익" in q:
        want = "손익"
    if not want:
        return None

    month_idx = _col_index(columns, ("기준월", "월"))
    dept_idx = _col_index(columns, ("부서",))
    acct_idx = _col_index(columns, ("계정과목", "과목"))
    target_idx = _col_index(columns, (want,))
    if None in (dept_idx, target_idx):
        return None

    acct_needle = None
    if "제품매출" in q:
        acct_needle = "제품매출"
    elif "제조원가" in q:
        acct_needle = "제조원가"
    elif "관리비" in q:
        acct_needle = "관리비"
    elif "운송비" in q:
        acct_needle = "운송비"

    for row in rows:
        if dept_idx >= len(row) or target_idx >= len(row):
            continue
        if dept not in str(row[dept_idx]):
            continue
        if acct_needle and acct_idx is not None and acct_idx < len(row):
            if acct_needle not in str(row[acct_idx]):
                continue
        if month_idx is not None and month_idx < len(row):
            mcell = str(row[month_idx]).strip()
            if month not in mcell:
                continue
        val = row[target_idx]
        return "\n".join([summary, f"{month} {dept} ({acct_needle or '해당 행'}) {want}: {val}"])
    return None


OPERATION_RUNNERS: tuple[Callable[..., str | None], ...] = (
    _try_finance_metric,
    _try_production_cell,
    _try_sum_column_by_date,
    _try_max_column,
    _try_count_unit_price_equals,
    _try_count_category_value,
    _try_count_column_equals_number,
    _try_count_unit_equals,
    _try_count_date_column_equals,
    _try_count_manager_items,
    _try_min_unit_price_in_category,
)


def run_table_operations(query: str, label: str, table_data: dict | None) -> str | None:
    """
    등록된 연산을 순서대로 시도한다. 하나라도 성공하면 답변 문자열을 반환한다.
    label은 향후 도메인 제한에 사용할 수 있어 시그니처만 유지한다.
    """
    _ = label
    if not table_data or not table_data.get("rows"):
        return None
    columns = table_data.get("columns") or []
    rows = table_data.get("rows") or []
    summary = (table_data.get("summary") or "").strip() or "조회 결과입니다."

    for runner in OPERATION_RUNNERS:
        try:
            out = runner(query, columns, rows, summary)
            if out:
                return out
        except Exception:
            continue
    return None
