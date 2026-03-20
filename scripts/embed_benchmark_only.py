#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
순정 임베딩 성능만 측정하는 벤치마크.

- 토큰 행 매칭, 수치 조건(column_hint), 기간 필터, 라우터 등은 사용하지 않습니다.
- 각 CSV 행을 "컬럼명 + 값" 한 덩어리 텍스트로 만든 뒤, 질문과 코사인 유사도만 계산합니다.

사용 예 (프로젝트 루트에서):

  conda run -n llm python scripts/embed_benchmark_only.py --query "단가 5만원 넘는 제품"
  conda run -n llm python scripts/embed_benchmark_only.py --query "출근시간" --top-k 20
  conda run -n llm python scripts/embed_benchmark_only.py --file data/inventory.csv --query "재고"

환경변수 DATA_DIR 이 있으면 해당 폴더를 스캔합니다 (없으면 프로젝트의 data/).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

# 프로젝트 루트 (scripts/ 의 부모)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL = "nlpai-lab/KURE-v1"

# data_reader.IGNORE_FILENAMES 와 동기화 (data_reader import 안 함 — 모델 이중 로딩 방지)
IGNORE_FILENAMES = {
    "random_questions_1000.csv",
    "random_questions_1000.jsonl",
    "clarified_training_dataset.jsonl",
}


def _read_csv_simple(path: str, max_rows: int) -> tuple[list[str], list[dict[str, str]]]:
    """인코딩만 맞춰 CSV를 dict 리스트로 읽습니다."""
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                first = next(reader, None)
                if not first:
                    return [], []
                headers = [str(c).strip() or f"col{i}" for i, c in enumerate(first)]
                # 중복 헤더 이름 고유화
                seen: dict[str, int] = {}
                uniq: list[str] = []
                for h in headers:
                    base = h
                    name = h
                    n = 2
                    while name in seen:
                        name = f"{base}_{n}"
                        n += 1
                    seen[name] = 1
                    uniq.append(name)
                headers = uniq
                rows: list[dict[str, str]] = []
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    cells = list(row)
                    while len(cells) < len(headers):
                        cells.append("")
                    rows.append(
                        {headers[j]: (str(cells[j]).strip() if j < len(cells) else "") for j in range(len(headers))}
                    )
                return headers, rows
        except (UnicodeDecodeError, UnicodeError):
            continue
    return [], []


def row_to_embed_text(headers: list[str], row: dict[str, str]) -> str:
    """임베딩 입력용: 컬럼명과 값을 모두 넣어 의미가 드러나게 합니다."""
    parts: list[str] = []
    for h in headers:
        v = row.get(h, "")
        parts.append(f"{h} {v}")
    return " ".join(parts)


def cosine_scores(query_emb: np.ndarray, row_embs: np.ndarray) -> np.ndarray:
    """query_emb (d,), row_embs (n, d) -> (n,) 코사인 유사도."""
    qn = np.linalg.norm(query_emb)
    rn = np.linalg.norm(row_embs, axis=1)
    qn = max(qn, 1e-10)
    rn = np.maximum(rn, 1e-10)
    q = query_emb / qn
    r = row_embs / rn[:, np.newaxis]
    return r @ q


def collect_csv_paths(data_dir: str, single_file: str | None) -> list[str]:
    if single_file:
        p = os.path.abspath(single_file)
        if not os.path.isfile(p):
            print(f"파일 없음: {p}", file=sys.stderr)
            sys.exit(1)
        return [p]
    patterns = [
        os.path.join(data_dir, "*.csv"),
        os.path.join(data_dir, "**", "*.csv"),
    ]
    found: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.basename(path) in IGNORE_FILENAMES:
                continue
            found.add(os.path.abspath(path))
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="순정 임베딩만으로 질문-행 Top-K 유사도")
    parser.add_argument("--query", "-q", required=True, help="테스트 질문")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="SentenceTransformer 모델 ID")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data")),
        help="CSV 스캔 폴더 (기본: DATA_DIR 또는 ./data)",
    )
    parser.add_argument("--file", "-f", default=None, help="단일 CSV만 사용할 때 경로")
    parser.add_argument("--top-k", "-k", type=int, default=15, help="출력할 상위 개수")
    parser.add_argument("--max-rows-per-file", type=int, default=5000, help="파일당 최대 행 수 (메모리/시간)")
    parser.add_argument("--batch-size", type=int, default=64, help="encode 배치 크기")
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="파일마다 상위 k개를 따로 출력 (기본: 전체 파일 합쳐서 전역 Top-K)",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    paths = collect_csv_paths(args.data_dir, args.file)
    if not paths:
        print(f"CSV 없음: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[모델 로딩] {args.model}", flush=True)
    model = SentenceTransformer(args.model)
    print(f"[질문] {args.query!r}", flush=True)
    print(f"[파일 수] {len(paths)}", flush=True)

    q_emb = model.encode(args.query, convert_to_numpy=True)

    if args.per_file:
        for path in paths:
            headers, rows = _read_csv_simple(path, args.max_rows_per_file)
            if not rows:
                print(f"\n--- {os.path.basename(path)} (행 없음) ---")
                continue
            texts = [row_to_embed_text(headers, r) for r in rows]
            embs = model.encode(texts, convert_to_numpy=True, batch_size=args.batch_size, show_progress_bar=False)
            scores = cosine_scores(q_emb, embs)
            order = np.argsort(-scores)[: args.top_k]
            print(f"\n=== {os.path.basename(path)} (행 {len(rows)}) ===")
            for rank, i in enumerate(order, 1):
                i = int(i)
                preview = texts[i][:200] + ("…" if len(texts[i]) > 200 else "")
                print(f"  {rank:2}. sim={scores[i]:.4f}  {preview}")
        return

    # 전역 Top-K: 모든 행에 점수 매기고 상위만
    all_scores: list[tuple[float, str, int, str]] = []
    for path in paths:
        headers, rows = _read_csv_simple(path, args.max_rows_per_file)
        if not rows:
            continue
        texts = [row_to_embed_text(headers, r) for r in rows]
        embs = model.encode(texts, convert_to_numpy=True, batch_size=args.batch_size, show_progress_bar=False)
        scores = cosine_scores(q_emb, embs)
        base = os.path.basename(path)
        for i in range(len(rows)):
            all_scores.append((float(scores[i]), base, i, texts[i]))

    all_scores.sort(key=lambda x: -x[0])
    print(f"\n[전역 Top-{args.top_k}] (총 후보 행 {len(all_scores)})\n")
    for rank, (sim, fname, ridx, text) in enumerate(all_scores[: args.top_k], 1):
        preview = text[:220] + ("…" if len(text) > 220 else "")
        print(f"  {rank:2}. sim={sim:.4f}  [{fname} row={ridx}]")
        print(f"      {preview}\n")


if __name__ == "__main__":
    main()
