import json
from app.db import get_cursor, now_iso


def log_action(employee_id, action_type, target_type="", target_id="", payload=None):
    payload_json = json.dumps(payload or {}, ensure_ascii=False)
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO audit_logs (employee_id, action_type, target_type, target_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (employee_id or "anonymous", action_type, target_type, str(target_id), payload_json, now_iso()),
        )


def list_audit_logs(limit=100):
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

