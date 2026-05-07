# load_excel.py

import pandas as pd
import sqlite3
import os


EXCEL_PATH = "data/inventory.xlsx"
DB_PATH = "db/excel_to_db.db"
TABLE_NAME = "inventory"


# ---------------- Excel → SQLite ----------------

def load_excel_to_sqlite():

    print("... Excel 읽는 중 ...")

    df = pd.read_excel(EXCEL_PATH)


    # 날짜 변환
    df["입고일"] = pd.to_datetime(df["입고일"])
    df["최근출고일"] = pd.to_datetime(df["최근출고일"])

    # 숫자형 변환
    numeric_columns = [
        "재고수량",
        "안전재고",
        "단가",
        "재고금액"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    os.makedirs("db", exist_ok=True)

    print("... SQLite 저장 중 ...")

    conn = sqlite3.connect(DB_PATH)

    df.to_sql(
        TABLE_NAME,
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    print("✅ SQLite 적재 완료")

