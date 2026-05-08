# utils/logger.py

import os
import json

from datetime import datetime


LOG_PATH = "logs/chat.log"


# ---------------- 로그 저장 ----------------

def save_log(

    query,
    ast,
    validation,
    result,
    response_time

):

    # logs 폴더 생성
    os.makedirs(
        "logs",
        exist_ok=True
    )

    # ---------------- 로그 데이터 ----------------

    log_data = {

        "질문일시": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "질문": query,

        "AST": ast,

        "VALIDATION": validation,

        "응답시간(초)": round(
            response_time,
            2
        ),

        "응답": result
    }

    # ---------------- JSONL 저장 ----------------

    with open(

        LOG_PATH,
        "a",
        encoding="utf-8"

    ) as f:

        f.write(

            json.dumps(
                log_data,
                ensure_ascii=False
            )
        )

        f.write("\n")
