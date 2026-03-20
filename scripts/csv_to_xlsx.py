"""data 폴더의 CSV를 모두 xlsx로 변환합니다. 한글 인코딩 유지.
프로젝트 루트에서 실행: python scripts/csv_to_xlsx.py
"""
import csv
import os
import sys
from pathlib import Path

try:
    import openpyxl
    from openpyxl import Workbook
except ImportError:
    print("openpyxl 필요: pip install openpyxl")
    raise

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
DATA_DIR = ROOT / "data"
ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


def read_csv(path: Path) -> list[list]:
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.reader(f))
        except Exception:
            continue
    raise ValueError(f"읽기 실패: {path}")


def write_xlsx(path: Path, rows: list[list]) -> None:
    wb = Workbook()
    ws = wb.active
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)
    out = path.with_suffix(".xlsx")
    wb.save(out)
    print(f"  → {out.name}")


def main(delete_csv: bool = False):
    if not DATA_DIR.exists():
        print(f"폴더 없음: {DATA_DIR}")
        return
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("변환할 CSV 없음.")
        return
    print(f"변환 대상: {len(csv_files)}개")
    for path in csv_files:
        if path.name.startswith("."):
            continue
        try:
            rows = read_csv(path)
            if not rows:
                print(f"  건너뜀(빈 파일): {path.name}")
                continue
            write_xlsx(path, rows)
            if delete_csv:
                path.unlink()
                print(f"  삭제: {path.name}")
        except Exception as e:
            print(f"  실패 {path.name}: {e}")
    print("완료.")


if __name__ == "__main__":
    delete = "--delete-csv" in sys.argv
    if delete:
        print("변환 후 CSV 파일 삭제 모드")
    main(delete_csv=delete)
