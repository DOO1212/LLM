import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime


DB_PATH = os.environ.get("APP_DB_PATH", "corpdesk.db")


def _dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = _dict_factory
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_cursor(commit=False):
    conn = get_connection()
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    finally:
        cur.close()
        conn.close()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def init_db():
    for attempt in range(3):
        try:
            with get_cursor(commit=True) as cur:
                cur.executescript(
                    """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                department TEXT DEFAULT '',
                position TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                target_type TEXT DEFAULT '',
                target_id TEXT DEFAULT '',
                payload_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS approval_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_type TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                drafter_employee_id TEXT NOT NULL,
                current_step INTEGER NOT NULL DEFAULT 0,
                source_query TEXT DEFAULT '',
                source_answer TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS approval_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                step_order INTEGER NOT NULL,
                approver_employee_id TEXT NOT NULL,
                decision TEXT NOT NULL DEFAULT 'pending',
                decided_at TEXT DEFAULT '',
                comment TEXT DEFAULT '',
                FOREIGN KEY(document_id) REFERENCES approval_documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS board_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                board_type TEXT NOT NULL DEFAULT 'notice',
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                author_employee_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );

            DROP TABLE IF EXISTS attendance_records;
            DROP TABLE IF EXISTS vacations;
            DROP TABLE IF EXISTS schedules;
            DROP TABLE IF EXISTS notifications;
                    """
                )
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == 2:
                raise
            time.sleep(0.2 * (attempt + 1))


def seed_defaults():
    with get_cursor(commit=True) as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM employees")
        if cur.fetchone()["cnt"] == 0:
            created_at = now_iso()
            cur.executemany(
                """
                INSERT INTO employees (employee_id, name, role, department, position, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("admin", "관리자", "admin", "경영지원", "팀장", created_at),
                    ("manager1", "매니저", "manager", "생산", "팀장", created_at),
                    ("user1", "사용자", "user", "재고", "사원", created_at),
                ],
            )


