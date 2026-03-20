"""
CSV 행 임베딩 사전 계산 캐시 (디스크).

- 소스 파일 mtime, EMBEDDING_MODEL, text_cols, 행 수가 일치할 때만 로드합니다.
- 캐시 파일은 DATA_DIR(또는 CSV가 있는 폴더) 아래 .embedding_cache/ 에 저장합니다.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np

CACHE_VERSION = 1
CACHE_SUBDIR = ".embedding_cache"


def _path_fingerprint(abs_path: str) -> str:
    """동일 파일명 충돌 방지용 짧은 해시."""
    h = hashlib.sha256(os.path.normpath(abs_path).encode("utf-8")).hexdigest()
    return h[:10]


def cache_directory_for_file(csv_abs_path: str) -> str:
    """CSV와 같은 트리의 .embedding_cache (없으면 생성)."""
    parent = os.path.dirname(os.path.abspath(csv_abs_path))
    d = os.path.join(parent, CACHE_SUBDIR)
    os.makedirs(d, mode=0o755, exist_ok=True)
    return d


def cache_paths(csv_abs_path: str) -> tuple[str, str]:
    abs_p = os.path.abspath(csv_abs_path)
    base = os.path.basename(abs_p)
    fp = _path_fingerprint(abs_p)
    safe = f"{base}.{fp}"
    d = cache_directory_for_file(abs_p)
    npz = os.path.join(d, f"{safe}.row_emb.npz")
    meta = os.path.join(d, f"{safe}.row_emb.meta.json")
    return npz, meta


def save_row_embedding_cache(
    csv_abs_path: str,
    embeddings: np.ndarray,
    text_cols: list[str],
    model_id: str,
) -> tuple[str, str]:
    """
    embeddings: shape (n_rows, dim), float32 권장.
    반환: (npz_path, meta_path)
    """
    abs_p = os.path.abspath(csv_abs_path)
    mtime = os.path.getmtime(abs_p)
    npz_path, meta_path = cache_paths(abs_p)

    arr = np.asarray(embeddings, dtype=np.float32)
    np.savez_compressed(npz_path, embeddings=arr)

    meta: dict[str, Any] = {
        "version": CACHE_VERSION,
        "source_path": abs_p,
        "source_mtime": mtime,
        "model_id": model_id,
        "text_cols": list(text_cols),
        "num_rows": int(arr.shape[0]),
        "embedding_dim": int(arr.shape[1]) if arr.ndim == 2 else 0,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return npz_path, meta_path


def load_row_embedding_cache(
    csv_abs_path: str,
    text_cols: list[str],
    model_id: str,
) -> np.ndarray | None:
    """
    유효하면 (n_rows, dim) float32 임베딩 행렬, 아니면 None.
    """
    abs_p = os.path.abspath(csv_abs_path)
    npz_path, meta_path = cache_paths(abs_p)

    if not os.path.isfile(npz_path) or not os.path.isfile(meta_path):
        return None

    try:
        current_mtime = os.path.getmtime(abs_p)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if int(meta.get("version", 0)) != CACHE_VERSION:
        return None
    if str(meta.get("model_id", "")) != str(model_id):
        return None
    if float(meta.get("source_mtime", -1)) != float(current_mtime):
        return None
    if list(meta.get("text_cols") or []) != list(text_cols):
        return None

    try:
        data = np.load(npz_path)
        emb = np.asarray(data["embeddings"], dtype=np.float32)
    except (OSError, KeyError, ValueError):
        return None

    n_expected = int(meta.get("num_rows", -1))
    if n_expected >= 0 and emb.shape[0] != n_expected:
        return None

    return emb
