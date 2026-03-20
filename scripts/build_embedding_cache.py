#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 전체 행에 대한 문서 임베딩을 미리 계산해 디스크에 저장합니다.
data_reader._filter_rows 가 토큰이 없을 때 쓰는 것과 동일한 text_cols / 인코딩 규칙을 따릅니다.

프로젝트 루트에서:

  python scripts/build_embedding_cache.py --file data/inventory.csv
  python scripts/build_embedding_cache.py --all-csv        # DATA_DIR 내 모든 CSV

환경변수:
  DATA_DIR          데이터 폴더 (기본: 프로젝트 ./data)
  EMBEDDING_MODEL   data_reader와 동일한 모델 ID (기본: nlpai-lab/KURE-v1)

CSV를 수정한 뒤에는 이 스크립트를 다시 실행하세요 (mtime이 바뀌면 기존 캐시는 자동 무효).
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# data_reader import 시 임베딩 모델 로딩됨
from app.services.embedding_cache import save_row_embedding_cache  # noqa: E402
from data_reader import (  # noqa: E402
    IGNORE_FILENAMES,
    WATCH_DIR,
    _EMBEDDING_MODEL_ID,
    _encode_docs,
    _embedding_text_columns,
    _read_file,
)


def _iter_csv_paths(data_dir: str) -> list[str]:
    import glob

    found: set[str] = set()
    for pattern in (
        os.path.join(data_dir, "*.csv"),
        os.path.join(data_dir, "**", "*.csv"),
    ):
        for p in glob.glob(pattern, recursive=True):
            if os.path.basename(p) in IGNORE_FILENAMES:
                continue
            found.add(os.path.abspath(p))
    return sorted(found)


def build_one(csv_path: str, batch_size: int) -> bool:
    abs_p = os.path.abspath(csv_path)
    rows = _read_file(abs_p)
    if not rows:
        print(f"[건너뜀] 행 없음: {abs_p}")
        return False
    columns = list(rows[0].keys())
    text_cols = _embedding_text_columns(columns, rows)
    if not text_cols:
        print(f"[건너뜀] 임베딩용 텍스트 컬럼 없음: {abs_p}")
        return False
    texts = [" ".join(str(row.get(c, "")) for c in text_cols) for row in rows]
    print(f"[인코딩] {os.path.basename(abs_p)}  행={len(texts)}  text_cols={text_cols}  model={_EMBEDDING_MODEL_ID}")
    embs = _encode_docs(texts, batch_size=batch_size, show_progress_bar=True)
    npz, meta = save_row_embedding_cache(abs_p, embs, text_cols, _EMBEDDING_MODEL_ID)
    print(f"[저장] {npz}")
    print(f"       {meta}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 행 임베딩 사전 계산 (data_reader 캐시)")
    parser.add_argument("--file", "-f", default=None, help="단일 CSV 경로")
    parser.add_argument(
        "--all-csv",
        action="store_true",
        help=f"{WATCH_DIR} 아래 모든 CSV에 대해 빌드",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="encode 배치 크기")
    args = parser.parse_args()

    data_dir = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))

    if args.all_csv:
        paths = _iter_csv_paths(data_dir)
    elif args.file:
        paths = [os.path.abspath(args.file)]
        if not os.path.isfile(paths[0]):
            print(f"파일 없음: {paths[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.error("--file 또는 --all-csv 를 지정하세요.")

    if not paths:
        print("대상 CSV 없음", file=sys.stderr)
        sys.exit(1)

    ok = 0
    for p in paths:
        if build_one(p, args.batch_size):
            ok += 1
    print(f"[완료] {ok}/{len(paths)} 파일 처리")


if __name__ == "__main__":
    main()
