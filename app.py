from fastapi import FastAPI
from pydantic import BaseModel

import time

from load_excel import load_excel_to_sqlite
from request_parser import parse_query
from validator import validate_ast
from query_executor import route
from utils.logger import save_log

app = FastAPI()

DB_PATH = "db/excel_to_db.db"


# ---------------- Request DTO ----------------

class QuestionRequest(BaseModel):

    question: str


# ---------------- DB 초기화 ----------------

@app.on_event("startup")
def startup():

    initialize_database()


def initialize_database():

    if os.path.exists(DB_PATH):

        print("✅ SQLite DB 존재")

    else:

        print("⚠️ SQLite DB 없음")
        print("📦 Excel → SQLite 적재 시작")

        load_excel_to_sqlite()

        print("✅ DB 초기화 완료")


# ---------------- API ----------------

@app.post("/ask")
def ask_question(
        request: QuestionRequest
):

    query = request.question

    start_time = time.time()


    # ---------------- Parser ----------------

    ast = parse_query(query)


    # ---------------- Validator ----------------

    validation = validate_ast(ast)


    # ---------------- Validation 실패 ----------------

    if not validation["valid"]:

        response_time = (
            time.time() - start_time
        )

        save_log(

            query=query,

            ast=ast,

            validation=validation,

            result="VALIDATION_FAILED",

            response_time=response_time,

            semantics={},

            prompt_modules=[],

            sql=""
        )

        return {

            "ast": ast,

            "answer": "VALIDATION_FAILED"
        }


    # ---------------- Execute ----------------

    result = route(ast)


    # ---------------- 응답 시간 ----------------

    response_time = (
        time.time() - start_time
    )


    # ---------------- 로그 저장 ----------------

    save_log(

        query=query,

        ast=ast,

        validation=validation,

        result=result,

        response_time=response_time,

        semantics={},

        prompt_modules=[],

        sql=""
    )


    return {

        "ast": ast,

        "answer": result
    }