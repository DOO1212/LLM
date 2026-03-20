#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
샘플 데이터(CSV)에 대한 LLM 평가 세트 실행기.

  python scripts/run_llm_eval.py
  python scripts/run_llm_eval.py -m qwen2.5:7b --questions scripts/llm_eval_questions_50.json

기본값: 행 수·글자 수 제한 없음(전체 파일). 제한하려면 --max-rows N, --max-chars M (양수).

합격 기준:
- must_contain_all: 모두 포함(영문 대소문자 무시).
- must_contain_one_of(선택): 위를 만족한 뒤, 목록 중 하나 이상 포함.
- must_contain_all 생략 시 []로 처리.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_ask_module():
    path = _ROOT / "scripts" / "ask_file_ollama.py"
    spec = importlib.util.spec_from_file_location("ask_file_ollama", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


load_as_text = _load_ask_module().load_as_text


def call_ollama_generate(host: str, model: str, prompt: str, timeout: int = 600) -> str:
    """연결 실패 시 프로세스 종료하지 않고 예외를 던집니다."""
    url = host.rstrip("/") + "/api/generate"
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False},
        ensure_ascii=False,
    ).encode("utf-8")
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


def _norm(s: str) -> str:
    return s.casefold()


def grade(
    answer: str,
    must_all: list[str],
    must_one_of: list[str] | None = None,
) -> tuple[bool, list[str]]:
    a = _norm(answer)
    missing = [k for k in must_all if _norm(k) not in a]
    if missing:
        return False, missing
    if must_one_of:
        if not any(_norm(opt) in a for opt in must_one_of):
            return False, [f"다음 중 하나 필요: {', '.join(must_one_of)}"]
    return True, []


def main() -> None:
    p = argparse.ArgumentParser(description="LLM 평가 세트 (Ollama)")
    p.add_argument(
        "--questions",
        type=Path,
        default=_ROOT / "scripts" / "llm_eval_questions_50.json",
        help="질문 JSON 경로",
    )
    p.add_argument("-m", "--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
    p.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0이면 전체 행(잘라내지 않음). 양수면 앞에서 N행만.",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="0이면 길이 제한 없음. 양수면 프롬프트 데이터를 이 글자 수로 자름.",
    )
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--show-answers", action="store_true", help="각 답변 전체 출력")
    args = p.parse_args()

    data = json.loads(args.questions.read_text(encoding="utf-8"))
    system_hint = """다음은 사용자가 제공한 파일에서 읽은 데이터입니다.
- 답은 반드시 한국어로만 하세요.
- 데이터에 없는 내용은 지어내지 말고 "자료에 없음"이라고 하세요.
- 숫자·조건 질문은 표의 값을 근거로 계산·필터해 답하세요.
"""
    passed = 0
    rows_out: list[dict] = []

    for item in data:
        qid = item["id"]
        rel = item["file"]
        question = item["question"]
        must = item.get("must_contain_all") or []
        one_of = item.get("must_contain_one_of")
        path = _ROOT / rel
        if not path.is_file():
            print(f"[SKIP] {qid}: 파일 없음 {path}", file=sys.stderr)
            continue

        max_rows = None if args.max_rows <= 0 else args.max_rows
        data_text = load_as_text(str(path), None, max_rows)
        if args.max_chars > 0 and len(data_text) > args.max_chars:
            data_text = data_text[: args.max_chars] + "\n\n[알림: 길이 제한으로 이후 내용은 잘렸습니다.]\n"

        prompt = f"{system_hint}\n--- 데이터 시작 ---\n{data_text}\n--- 데이터 끝 ---\n\n질문: {question}\n"
        try:
            answer = call_ollama_generate(args.host, args.model, prompt, timeout=args.timeout)
        except Exception as e:
            answer = f"[오류] {e}"

        ok, missing = grade(answer, must, one_of)
        if ok:
            passed += 1
        row = {"id": qid, "pass": ok, "missing": missing, "answer_preview": answer[:240]}
        rows_out.append(row)

        status = "PASS" if ok else "FAIL"
        print(f"{status}  {qid}  {rel}")
        if not ok:
            print(f"       부족 키워드: {missing}")
        if args.show_answers:
            print(answer)
            print("-" * 60)

    total = len(rows_out)
    print()
    print(f"요약: {passed}/{total} 통과  (모델={args.model})")
    if total:
        print(f"통과율: {100.0 * passed / total:.1f}%")

    out_json = _ROOT / "scripts" / "llm_eval_last_result.json"
    out_json.write_text(
        json.dumps(
            {"model": args.model, "passed": passed, "total": total, "items": rows_out},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"결과 저장: {out_json}")


if __name__ == "__main__":
    main()
