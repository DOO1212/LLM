#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data 폴더 안의 모든 .xlsx 파일을 같은 이름의 .csv로 변환합니다.
시트가 여러 개면 시트명(또는 인덱스)을 붙인 파일명으로 저장합니다.

사용: python3 scripts/xlsx_to_csv.py
     python3 scripts/xlsx_to_csv.py /path/to/data
     python3 scripts/xlsx_to_csv.py --list   # data 폴더 내용만 출력
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    list_only = "--list" in sys.argv

    root = Path(__file__).resolve().parent.parent / "data"
    if args:
        root = Path(args[0]).resolve()

    if not root.is_dir():
        print(f"디렉터리가 없습니다: {root}", file=sys.stderr)
        sys.exit(1)

    if list_only:
        print(f"경로: {root}\n파일 목록:")
        for f in sorted(root.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(root)}")
        xlsx_files = [p for p in root.rglob("*") if p.suffix.lower() == ".xlsx"]
        print(f"\n.xlsx 파일 수: {len(xlsx_files)}")
        for p in xlsx_files:
            print(f"  {p.relative_to(root)}")
        return

    xlsx_files = list(root.glob("**/*.xlsx")) or [
        p for p in root.rglob("*") if p.suffix.lower() == ".xlsx"
    ]
    if not xlsx_files:
        print(f"{root} 안에 .xlsx 파일이 없습니다.")
        return

    for path in sorted(xlsx_files):
        try:
            xl = pd.ExcelFile(path)
            base = path.stem
            parent = path.parent

            for i, sheet_name in enumerate(xl.sheet_names):
                df = pd.read_excel(path, sheet_name=sheet_name)
                if len(xl.sheet_names) == 1:
                    out_path = parent / f"{base}.csv"
                else:
                    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(sheet_name))
                    out_path = parent / f"{base}_{safe_name}.csv"
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(out_path)
        except Exception as e:
            print(f"오류 {path}: {e}", file=sys.stderr)

    print(f"총 {len(xlsx_files)}개 xlsx 처리 완료.")


if __name__ == "__main__":
    main()
