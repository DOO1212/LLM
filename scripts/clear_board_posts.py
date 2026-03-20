#!/usr/bin/env python3
"""게시판(board_posts) 게시글 전부 삭제."""
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("APP_DB_PATH", os.path.join(ROOT, "corpdesk.db"))

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM board_posts")
    n = cur.rowcount
    conn.commit()
    conn.close()
    print(f"삭제 완료: 게시글 {n}건")

if __name__ == "__main__":
    main()
