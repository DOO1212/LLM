# app.py

from fastapi import FastAPI
from pydantic import BaseModel

from request_parser import parse_query
from validator import validate_ast
from query_builder import build_sql
from database import execute_query


# ---------------- FastAPI ----------------

app = FastAPI()


# ---------------- Request DTO ----------------

class QuestionRequest(BaseModel):

    question: str


# ---------------- Response DTO ----------------

class AnswerResponse(BaseModel):

    ast: dict

    result: object


# ---------------- API ----------------

@app.post("/ask", response_model=AnswerResponse)

def ask_question(request: QuestionRequest):

    query = request.question


    # ---------------- AST 생성 ----------------

    ast = parse_query(query)


    # ---------------- Validation ----------------

    validation = validate_ast(ast)


    if not validation["valid"]:

        return {

            "ast": ast,

            "result": {

                "error": validation["reason"]
            }
        }


    # ---------------- SQL 생성 ----------------

    sql = build_sql(ast)


    # ---------------- SQL 실행 ----------------

    result = execute_query(sql)


    # ---------------- Response ----------------

    return {

        "ast": ast,

        "result": result
    }