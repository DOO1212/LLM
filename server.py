from fastapi import FastAPI
from pydantic import BaseModel

from request_parser import parse_query
from router import route
from output_template import format_output

app = FastAPI()


# 요청 데이터 형식
class QueryRequest(BaseModel):
    query: str


# 서버 시작 시 1번 실행
@app.on_event("startup")
def load_model():
    from request_parser import get_model
    get_model()
    print("✅ 모델 로딩 완료")


# API 엔드포인트
@app.post("/query")
def query_api(req: QueryRequest):

    parsed = parse_query(req.query)
    result = route(parsed)
    output = format_output(result)

    return {
        "query": req.query,
        "parsed": parsed,
        "result": output
    }
