#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pytest + 50문항 run_llm_eval 실행 후 scripts/test_run_report.txt 에 기록."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "scripts" / "test_run_report.txt"


def run_step(title: str, args: list[str]) -> int:
    lines.append(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")
    p = subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    lines.append(p.stdout or "")
    lines.append(p.stderr or "")
    lines.append(f"\n[exit code {p.returncode}]\n")
    return p.returncode


lines: list[str] = []
lines.append(f"python: {sys.executable}\nROOT: {ROOT}\n")

rc1 = run_step("pytest tests/test_table_operations.py", ["-m", "pytest", "tests/test_table_operations.py", "-v", "--tb=short"])
model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
rc2 = run_step(f"run_llm_eval.py -m {model}", ["scripts/run_llm_eval.py", "-m", model])

lines.append(f"\n요약: pytest={rc1}, run_llm_eval={rc2}\n")
REPORT.write_text("".join(lines), encoding="utf-8")
print(f"Wrote {REPORT}")
sys.exit(0 if (rc1 == 0 and rc2 == 0) else 1)
