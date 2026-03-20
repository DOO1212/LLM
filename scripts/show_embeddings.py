#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
저장된(또는 즉시 계산한) CSV 행 임베딩 좌표를 터미널에 출력합니다.

  python scripts/show_embeddings.py --file data/inventory.csv
  python scripts/show_embeddings.py --file data/inventory.csv --rows 2 --dims 16

캐시가 있으면 .npz에서 읽고, 없으면 data_reader로 해당 파일만 인코딩합니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 행 임베딩 좌표 보기")
    parser.add_argument("--file", "-f", default="data/inventory.csv", help="CSV 경로")
    parser.add_argument("--rows", "-r", type=int, default=3, help="몇 행까지 출력 (기본 3)")
    parser.add_argument("--dims", "-d", type=int, default=20, help="벡터 앞쪽 몇 차원 출력 (기본 20)")
    args = parser.parse_args()

    abs_path = os.path.abspath(os.path.join(ROOT_DIR, args.file))
    if not os.path.isfile(abs_path):
        print(f"파일 없음: {abs_path}", file=sys.stderr)
        sys.exit(1)

    from app.services.embedding_cache import cache_paths, load_row_embedding_cache

    npz_path, meta_path = cache_paths(abs_path)
    emb = None
    text_cols = None
    if os.path.isfile(npz_path) and os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        text_cols = meta.get("text_cols") or []
        from data_reader import _EMBEDDING_MODEL_ID

        emb = load_row_embedding_cache(abs_path, text_cols, _EMBEDDING_MODEL_ID)
        if emb is not None:
            print(f"[캐시 사용] {npz_path}")
            print(f"  모델: {meta.get('model_id')}  |  행 수: {emb.shape[0]}  |  차원: {emb.shape[1]}")
            print()

    if emb is None:
        print("[캐시 없음] data_reader로 인코딩합니다 (모델 로딩 1회)...")
        from data_reader import (
            _EMBEDDING_MODEL_ID,
            _embedding_text_columns,
            _encode_docs,
            _read_file,
        )

        rows = _read_file(abs_path)
        if not rows:
            print("행 없음", file=sys.stderr)
            sys.exit(1)
        columns = list(rows[0].keys())
        text_cols = _embedding_text_columns(columns, rows)
        if not text_cols:
            print("임베딩용 텍스트 컬럼 없음", file=sys.stderr)
            sys.exit(1)
        texts = [" ".join(str(row.get(c, "")) for c in text_cols) for row in rows]
        import numpy as np

        emb = _encode_docs(texts, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        print(f"  모델: {_EMBEDDING_MODEL_ID}  |  행 수: {emb.shape[0]}  |  차원: {emb.shape[1]}")
        print()

    n_rows = min(args.rows, emb.shape[0])
    n_dims = min(args.dims, emb.shape[1])

    print("--- 임베딩 좌표 (앞쪽 차원만) ---")
    for i in range(n_rows):
        vec = emb[i]
        head = ", ".join(f"{vec[j]:.6f}" for j in range(n_dims))
        print(f"행 {i}: [{head}, ... ]  # shape (1, {emb.shape[1]})")
    print()
    print(f"전체 shape: ({emb.shape[0]}, {emb.shape[1]})")


if __name__ == "__main__":
    main()
