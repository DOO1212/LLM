import json
import sqlite3
from pathlib import Path


def main():
    conn = sqlite3.connect("file:corpdesk.db?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, payload_json FROM audit_logs WHERE action_type=? ORDER BY id DESC LIMIT 12",
        ("chatbot.ask",),
    )
    rows = cur.fetchall()
    out = []
    for r in rows:
        payload = json.loads(r["payload_json"] or "{}")
        evidence = payload.get("data_evidence")
        out.append("---")
        out.append(f"id={r['id']} created_at={r['created_at']}")
        out.append(f"query={payload.get('query')}")
        out.append(f"effective_query={payload.get('effective_query')}")
        out.append(f"final_label={payload.get('final_label')} status={payload.get('status')}")
        out.append(f"context_query_expanded={payload.get('context_query_expanded')}")
        out.append("data_evidence=" + json.dumps(evidence, ensure_ascii=False))
    conn.close()
    Path("audit_debug_latest.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
