# utils/logger.py

import os
import json

from datetime import datetime


LOG_PATH = "logs/chat.log"


# ---------------- Save Log ----------------

def save_log(

    query,
    semantics,
    prompt_modules,
    ast,
    validation,
    sql,
    result,
    response_time
):

    # ---------------- logs 폴더 생성 ----------------

    os.makedirs(

        "logs",
        exist_ok=True
    )


    # ---------------- Log Data ----------------

    log_data = {

        "질문일시": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "질문": query,

        "Semantics": semantics,

        "PromptModules": prompt_modules,

        "AST": ast,

        "Validation": validation,

        "SQL": sql,

        "응답": result,

        "응답시간(초)": round(
            response_time,
            2
        )
    }


    # ---------------- Save ----------------

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
