"""백엔드 표 연산 레지스트리 스모크 테스트."""
import csv
from pathlib import Path

import pytest

from app.services.table_operations import run_table_operations

ROOT = Path(__file__).resolve().parents[1]


def _load_csv(rel: str) -> tuple[list[str], list[list]]:
    path = ROOT / rel
    with open(path, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        headers = next(r)
        rows = []
        for row in r:
            row = list(row)
            while len(row) < len(headers):
                row.append("")
            rows.append(row[: len(headers)])
    return headers, rows


def _td(headers, rows):
    return {"columns": headers, "rows": rows, "summary": "요약"}


def test_count_unit_price_equals():
    h, rows = _load_csv("data/inventory.csv")
    out = run_table_operations("단가(원)가 정확히 98000원인 행은 몇 건인가?", "재고", _td(h, rows))
    assert out is not None
    assert "2" in out


def test_sum_defects_by_date():
    h, rows = _load_csv("data/production.csv")
    out = run_table_operations("2026-03-10 하루 전체의 불량수 합계", "생산", _td(h, rows))
    assert out is not None
    assert "160" in out


def test_finance_vouchers():
    h, rows = _load_csv("data/finance.csv")
    out = run_table_operations("2026-03 관리 부서의 전표처리건수는?", "재무", _td(h, rows))
    assert out is not None
    assert "61" in out


def test_min_price_hydraulic():
    h, rows = _load_csv("data/inventory.csv")
    out = run_table_operations("유압/공압 카테고리에서 단가가 가장 낮은 품목명은?", "재고", _td(h, rows))
    assert out is not None
    assert "유압 호스" in out
