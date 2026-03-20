#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로컬 파일(CSV/xlsx)을 읽어 Ollama에 붙인 뒤 질문에 답하게 합니다.
모델이 디스크를 직접 읽는 게 아니라, 이 스크립트가 읽어서 프롬프트로 넘깁니다.

사용 예:
  export OLLAMA_HOST=http://127.0.0.1:11434   # 기본값
  python3 scripts/ask_file_ollama.py /workspace/inventory.xlsx "10만원 이상 제품은?" -m qwen2.5:7b

의존성: pandas, openpyxl(xlsx용)
  pip install pandas openpyxl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def load_as_text(path: str, sheet: str | int | None, max_rows: int | None) -> str:
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        kwargs: dict = {}
        if sheet is not None:
            kwargs["sheet_name"] = int(sheet) if str(sheet).isdigit() else sheet
        df = pd.read_excel(path, **kwargs)
    elif ext in (".csv", ".txt"):
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(path, encoding="utf-8", errors="replace")
    else:
        with open(path, "rb") as f:
            raw = f.read()
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return raw.decode("cp949", errors="replace")

    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
        note = f"\n\n[알림: 전체 중 앞쪽 {max_rows}행만 포함했습니다.]\n"
    else:
        note = ""

    # 표를 읽기 쉬운 텍스트로
    text = df.to_csv(index=False)
    return f"파일: {os.path.basename(path)}\n행 수(이 블록): {len(df)}\n\n{text}{note}"


def call_ollama(host: str, model: str, prompt: str, timeout: int = 600) -> str:
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
        print("Ollama 연결 실패:", e, file=sys.stderr)
        print("ollama serve 가 떠 있는지, OLLAMA_HOST 가 맞는지 확인하세요.", file=sys.stderr)
        sys.exit(1)
    return data.get("response", "")


def main() -> None:
    p = argparse.ArgumentParser(description="파일 내용을 읽어 Ollama에 질문합니다.")
    p.add_argument("file", help="CSV 또는 xlsx 경로")
    p.add_argument("question", help="질문 (따옴표로 감싸기)")
    p.add_argument("-m", "--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
    p.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama API 주소",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=100_000,
        help="프롬프트에 넣는 데이터 최대 글자 수 (넘치면 잘림)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="표는 최대 N행만 사용 (미지정이면 전체 후 max-chars로 자름)",
    )
    p.add_argument(
        "--sheet",
        default=None,
        help="xlsx 시트 이름 또는 번호(0부터)",
    )
    args = p.parse_args()

    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        print("파일 없음:", path, file=sys.stderr)
        sys.exit(1)

    data_text = load_as_text(path, args.sheet, args.max_rows)
    if len(data_text) > args.max_chars:
        data_text = data_text[: args.max_chars] + "\n\n[알림: 길이 제한으로 이후 내용은 잘렸습니다.]\n"

    system_hint = """다음은 사용자가 제공한 파일에서 읽은 데이터입니다.
- 답은 반드시 한국어로만 하세요.
- 데이터에 없는 내용은 지어내지 말고 "자료에 없음"이라고 하세요.
- 숫자·조건 질문은 표의 값을 근거로 계산·필터해 답하세요.
"""
    prompt = f"{system_hint}\n--- 데이터 시작 ---\n{data_text}\n--- 데이터 끝 ---\n\n질문: {args.question}\n"

    print("모델:", args.model, "| 데이터 글자 수:", len(data_text), file=sys.stderr)
    out = call_ollama(args.host, args.model, prompt)
    print(out)


if __name__ == "__main__":
    main()
