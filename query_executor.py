# router.py

import sqlite3
from query_builder import build_sql


# ---------------- 설정 ----------------

DB_PATH = "db/excel_to_db.db"


# ---------------- SQL 실행 ----------------

def execute_query(sql):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    print(f"[SQL]\n{sql}")

    cursor.execute(sql)

    rows = cursor.fetchall()

    # 컬럼명 추출
    columns = [
        desc[0]
        for desc in cursor.description
    ]

    conn.close()

    return rows, columns


# ---------------- 결과 변환 ----------------

def format_result(rows, columns):

    results = []

    for row in rows:

        results.append({
            col: value
            for col, value in zip(columns, row)
        })

    return results


# ---------------- 메인 Router ----------------

def route(ast):

    sql = build_sql(ast)

    rows, columns = execute_query(sql)

    results = format_result(rows, columns)

    return results