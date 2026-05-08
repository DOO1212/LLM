# utils/ast_cache.py

import json
import os


CACHE_PATH = "cache/ast_cache.json"


# ---------------- 캐시 로드 ----------------

def load_cache():

    os.makedirs(
        "cache",
        exist_ok=True
    )

    if not os.path.exists(CACHE_PATH):

        return {}


    try:

        with open(

            CACHE_PATH,
            "r",
            encoding="utf-8"

        ) as f:

            return json.load(f)


    except json.JSONDecodeError:

        print(
            "⚠️ AST 캐시 파일 손상 감지"
        )

        return {}


# ---------------- 캐시 저장 ----------------

def save_cache(cache):

    with open(

        CACHE_PATH,
        "w",
        encoding="utf-8"

    ) as f:

        json.dump(

            cache,
            f,

            ensure_ascii=False,
            indent=2
        )


# ---------------- AST 조회 ----------------

def get_cached_ast(query):

    cache = load_cache()

    return cache.get(query)


# ---------------- AST 저장 ----------------

def cache_ast(query, ast):

    cache = load_cache()

    cache[query] = ast

    save_cache(cache)