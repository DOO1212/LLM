# main.py

import os
import time

from load_excel import load_excel_to_sqlite
from request_parser import parse_query
from validator import validate_ast
from query_executor import route
from utils.logger import save_log


DB_PATH = "db/excel_to_db.db"


# ---------------- DB 초기화 ----------------

def initialize_database():

    if os.path.exists(DB_PATH):

        print("✅ SQLite DB 존재")

    else:

        print("⚠️ SQLite DB 없음")
        print("📦 Excel → SQLite 적재 시작")

        load_excel_to_sqlite()

        print("✅ DB 초기화 완료")


# ---------------- 결과 출력 ----------------

def print_result(result):

    print("\n[RESULT]")

    if not result:

        print("❌ 결과 없음")

        return

    for row in result:

        print(row)


# ---------------- 메인 ----------------

def main():

    # DB 준비
    initialize_database()

    while True:

        query = input("\n질문 (종료: exit): ").strip()

        # 종료
        if query.lower() == "exit":

            print("👋 프로그램 종료")

            break

        # --------------- 시작 시간 -------------

        start_time = time.time()


        # ---------------- Parser ----------------

        ast = parse_query(query)

        print("\n[AST]")
        print(ast)


        # ---------------- Validator ----------------

        validation = validate_ast(ast)

        print("\n[VALIDATION]")
        print(validation)


        # ---------------- Validation 실패 ----------------

        if not validation["valid"]:

            response_time = (
                time.time() - start_time
            )

            print("\n❌ 지원하지 않는 질문입니다.")
            print(f"사유: {validation['reason']}")


            # 로그 저장
            save_log(

                query=query,

                ast=ast,

                validation=validation,

                result="VALIDATION_FAILED",

                response_time=response_time
            )

            continue



        # ---------------- Router ----------------

        result = route(ast)



        # ---------------- 응답 시간 ----------------

        response_time = (
            time.time() - start_time
        )



        # ---------------- Result ----------------

        print_result(result)


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


if __name__ == "__main__":

    main()