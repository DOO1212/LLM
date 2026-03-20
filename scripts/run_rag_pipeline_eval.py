#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계 RAG 스모크/벤치: 임베딩 → 행 단위 유사도 검색 → LLM 답변

  1단계: data_reader와 동일한 임베딩 모델로 쿼리·각 행 텍스트 인코딩
  2단계: 코사인 유사도 상위 --top-k 행만 컨텍스트로 선택
  3단계: Ollama 또는 answer_llm.generate_answer 로 답 생성

단일 질문:
  python scripts/run_rag_pipeline_eval.py --file data/inventory.csv \\
    --query "단가 10만원 넘는 품목 알려줘" --top-k 8

50문항 배치(키워드 자동 채점):
  python scripts/run_rag_pipeline_eval.py --suite scripts/llm_eval_questions_50.json \\
    --top-k 10 -m qwen2.5:7b

주의: 표 데이터에서 검색이 빗나가면(관련 행이 top-k에 안 들어옴) 답이 틀릴 수 있습니다.

  --full-row-embed  임베딩 입력에 모든 컬럼(단가 등 숫자 열 포함)을 붙입니다.
      기본은 data_reader._embedding_text_columns 와 동일(숫자 위주 열 제외)이라
      "단가 N원" 질문이 검색 단계에서 놓치기 쉽습니다. 평가·실험용으로 사용하세요.
"""
from __future__ import annotations

import argparse
import numpy as np
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _norm(s: str) -> str:
    return s.casefold()


def grade(answer: str, must_all: list[str], must_one_of: list[str] | None = None) -> tuple[bool, list[str]]:
    a = _norm(answer)
    missing = [k for k in must_all if _norm(k) not in a]
    if missing:
        return False, missing
    if must_one_of:
        if not any(_norm(opt) in a for opt in must_one_of):
            return False, [f"다음 중 하나 필요: {', '.join(must_one_of)}"]
    return True, []


def call_ollama_generate(host: str, model: str, prompt: str, timeout: int = 600) -> str:
    url = host.rstrip("/") + "/api/generate"
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama 연결 실패: {e}") from e
    return str(data.get("response", ""))


def _load_data_reader():
    """프로젝트 루트에서 data_reader 임포트 (임베딩 모델 로드됨)."""
    path = _ROOT / "data_reader.py"
    spec = importlib.util.spec_from_file_location("data_reader", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _rows_to_llm_data(
    path: str,
    rows: list[dict],
    columns: list[str],
) -> dict:
    """answer_llm.generate_answer 에 넣을 data dict."""
    return {
        "label": "RAG",
        "filename": os.path.basename(path),
        "columns": columns,
        "rows": [[r.get(c, "") for c in columns] for r in rows],
    }


def _build_ollama_prompt(query: str, data: dict | None) -> str:
    if not data or not data.get("rows"):
        user = f"질문: {query}"
    else:
        cols = data["columns"]
        lines = []
        for row in data["rows"][:40]:
            pairs = []
            for c, v in zip(cols, row):
                if v in ("", None):
                    continue
                t = str(v).strip()
                if len(t) > 80:
                    t = t[:77] + "..."
                pairs.append(f"{c}: {t}")
            lines.append("- " + ", ".join(pairs))
        block = "\n".join(lines)
        user = (
            f"[검색으로 가져온 상위 행만 근거로 답하세요. 없으면 '데이터에 없습니다'.]\n"
            f"[{data.get('filename', '')}]\n{block}\n\n질문: {query}"
        )
    system = (
        "당신은 사내 업무 도우미입니다. 주어진 행만 근거로 한국어로 답하세요. "
        "근거에 없는 숫자는 만들지 마세요."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_pipeline(
    dr,
    csv_path: str,
    query: str,
    top_k: int,
    llm: str,
    ollama_host: str,
    ollama_model: str,
    *,
    full_row_embed: bool = False,
) -> dict:
    t0 = time.perf_counter()
    rows = dr._read_file(csv_path)
    if not rows:
        raise RuntimeError(f"행이 없습니다: {csv_path}")
    columns = list(rows[0].keys())
    if full_row_embed:
        text_cols_meta = columns
        row_texts = [
            " ".join(f"{c} {r.get(c, '')}" for c in columns)
            for r in rows
        ]
    else:
        text_cols_meta = dr._embedding_text_columns(columns, rows)
        if not text_cols_meta:
            raise RuntimeError("임베딩용 텍스트 컬럼이 없습니다.")
        row_texts = [" ".join(str(r.get(c, "")) for c in text_cols_meta) for r in rows]
    t_embed_start = time.perf_counter()
    q_emb = dr._encode_query(query.strip(), show_progress_bar=False)
    doc_embs = dr._encode_docs(row_texts, show_progress_bar=False)
    t_embed_done = time.perf_counter()

    q_emb = np.asarray(q_emb)
    if q_emb.ndim == 2:
        q_emb = q_emb.ravel()
    doc_embs = np.asarray(doc_embs)

    sims = [dr._cosine_sim(q_emb, doc_embs[i]) for i in range(len(rows))]
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    k = max(1, min(top_k, len(order)))
    picked = order[:k]
    retrieved = [rows[i] for i in picked]
    t_search_done = time.perf_counter()

    data = _rows_to_llm_data(csv_path, retrieved, columns)

    if llm == "ollama":
        prompt = _build_ollama_prompt(query, data)
        answer = call_ollama_generate(ollama_host, ollama_model, prompt)
    else:
        from answer_llm import generate_answer

        answer = generate_answer(query, "RAG", data)

    t_end = time.perf_counter()
    return {
        "query": query,
        "file": csv_path,
        "top_k": k,
        "text_cols": text_cols_meta,
        "full_row_embed": full_row_embed,
        "picked_indices": picked,
        "similarities": [round(sims[i], 4) for i in picked],
        "embed_seconds": round(t_embed_done - t_embed_start, 3),
        "search_seconds": round(t_search_done - t_embed_done, 3),
        "llm_seconds": round(t_end - t_search_done, 3),
        "total_seconds": round(t_end - t0, 3),
        "answer": answer,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="3단계 RAG 파이프라인 평가")
    p.add_argument("--file", help="CSV 경로 (단일 모드)")
    p.add_argument("--query", help="질문 (단일 모드)")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--llm",
        choices=("ollama", "local"),
        default="ollama",
        help="ollama=API, local=answer_llm (ANSWER_LLM_MODEL, 기본 7B)",
    )
    p.add_argument("-m", "--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--suite", type=Path, help="llm_eval_questions.json 배치 실행")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--full-row-embed",
        action="store_true",
        help="임베딩에 모든 컬럼 포함(단가 등 숫자 열 포함). 기본은 data_reader와 동일한 텍스트 열만.",
    )
    args = p.parse_args()

    os.chdir(_ROOT)
    sys.path.insert(0, str(_ROOT))

    print("[1/3] data_reader 로드(임베딩 모델)…", flush=True)
    dr = _load_data_reader()

    if args.suite:
        data = json.loads(Path(args.suite).read_text(encoding="utf-8"))
        passed = 0
        results = []
        for item in data:
            qid = item["id"]
            rel = item["file"]
            question = item["question"]
            must = item.get("must_contain_all") or []
            one_of = item.get("must_contain_one_of")
            path = str(_ROOT / rel)
            if not os.path.isfile(path):
                print(f"SKIP {qid} 파일 없음", flush=True)
                continue
            try:
                out = run_pipeline(
                    dr,
                    path,
                    question,
                    args.top_k,
                    args.llm,
                    args.host,
                    args.model,
                    full_row_embed=args.full_row_embed,
                )
            except Exception as e:
                results.append({"id": qid, "pass": False, "error": str(e)})
                print(f"FAIL {qid} {e}", flush=True)
                continue
            ok, missing = grade(out["answer"], must, one_of)
            if ok:
                passed += 1
            results.append({"id": qid, "pass": ok, "missing": missing, "rag": out})
            st = "PASS" if ok else "FAIL"
            print(f"{st}  {qid}  sim_top={out['similarities'][0] if out['similarities'] else '—'}", flush=True)
            if not ok:
                print(f"      {missing}", flush=True)

        total = len(results)
        print()
        print(
            f"[RAG 3단계] 통과 {passed}/{total}  top_k={args.top_k}  "
            f"full_row_embed={args.full_row_embed}  "
            f"llm={args.llm} ({args.model if args.llm == 'ollama' else 'answer_llm'})"
        )
        out_path = _ROOT / "scripts" / "rag_pipeline_last_result.json"
        out_path.write_text(
            json.dumps({"passed": passed, "total": total, "items": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"결과 저장: {out_path}")
        return

    if not args.file or not args.query:
        p.error("--file 과 --query 가 필요하거나 --suite 를 지정하세요.")

    out = run_pipeline(
        dr,
        os.path.abspath(args.file),
        args.query,
        args.top_k,
        args.llm,
        args.host,
        args.model,
        full_row_embed=args.full_row_embed,
    )
    print("=== 3단계 RAG 결과 ===")
    print(f"파일: {out['file']}")
    print(f"임베딩 컬럼: {out['text_cols']}")
    print(f"상위 유사도: {out['similarities'][:5]}{'...' if len(out['similarities']) > 5 else ''}")
    print(f"선택 행 인덱스(0-based 원본 순서): {out['picked_indices'][:10]}{'...' if len(out['picked_indices']) > 10 else ''}")
    print(f"시간: embed={out['embed_seconds']}s search={out['search_seconds']}s llm={out['llm_seconds']}s total={out['total_seconds']}s")
    print()
    print(out["answer"])


if __name__ == "__main__":
    main()
