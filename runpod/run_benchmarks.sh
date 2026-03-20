#!/usr/bin/env bash
# RunPod(또는 Linux GPU 머신)에서 벤치마크 일괄 실행.
# 사용: ./runpod/run_benchmarks.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/runpod/benchmark_output.txt"

export PYTHONUTF8="${PYTHONUTF8:-1}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"

{
  echo "========== benchmark $(date -Is) =========="
  echo "ROOT=$ROOT"
  echo "EMBEDDING_MODEL=${EMBEDDING_MODEL:-nlpai-lab/KURE-v1 (default)}"
  echo

  echo "========== nvidia-smi =========="
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "(no nvidia-smi)"
  echo

  echo "========== pip install =========="
  python3 -m pip install -q -r requirements.txt pytest
  echo

  echo "========== pytest tests/test_table_operations.py =========="
  python3 -m pytest tests/test_table_operations.py -v --tb=short
  echo

  echo "========== RAG 50q (embedding + search + answer_llm 3B local) =========="
  python3 scripts/run_rag_pipeline_eval.py \
    --suite scripts/llm_eval_questions_50.json \
    --top-k 10 \
    --llm local
  echo

  if command -v ollama >/dev/null 2>&1; then
    if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "========== run_llm_eval 50q (Ollama ${OLLAMA_MODEL}) =========="
      python3 scripts/run_llm_eval.py -m "${OLLAMA_MODEL}"
    else
      echo "========== SKIP run_llm_eval: Ollama not responding at ${OLLAMA_HOST} =========="
    fi
  else
    echo "========== SKIP run_llm_eval: ollama CLI not found =========="
  fi

  echo
  echo "========== done =========="
} 2>&1 | tee "$OUT"

echo
echo "Log saved: $OUT"
