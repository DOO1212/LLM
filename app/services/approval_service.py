from app.db import get_cursor, now_iso


def create_document(doc_type, title, body, drafter_employee_id, approvers=None, source_query="", source_answer=""):
    ts = now_iso()
    approvers = approvers or []
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO approval_documents
            (doc_type, title, body, status, drafter_employee_id, current_step, source_query, source_answer, created_at, updated_at)
            VALUES (?, ?, ?, 'draft', ?, 0, ?, ?, ?, ?)
            """,
            (doc_type, title, body, drafter_employee_id, source_query, source_answer, ts, ts),
        )
        doc_id = cur.lastrowid

        for idx, approver in enumerate(approvers, start=1):
            cur.execute(
                """
                INSERT INTO approval_steps (document_id, step_order, approver_employee_id, decision, decided_at, comment)
                VALUES (?, ?, ?, 'pending', '', '')
                """,
                (doc_id, idx, approver),
            )
    return doc_id


def submit_document(doc_id):
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE approval_documents
            SET status = 'pending', updated_at = ?
            WHERE id = ?
            """,
            (now_iso(), doc_id),
        )


def decide_document(doc_id, approver_employee_id, decision, comment=""):
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be approved or rejected")

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            SELECT * FROM approval_steps
            WHERE document_id = ? AND approver_employee_id = ? AND decision = 'pending'
            ORDER BY step_order ASC
            LIMIT 1
            """,
            (doc_id, approver_employee_id),
        )
        step = cur.fetchone()
        if not step:
            raise ValueError("처리 가능한 결재 단계가 없습니다.")

        cur.execute(
            """
            UPDATE approval_steps
            SET decision = ?, decided_at = ?, comment = ?
            WHERE id = ?
            """,
            (decision, now_iso(), comment, step["id"]),
        )

        if decision == "rejected":
            cur.execute(
                "UPDATE approval_documents SET status = 'rejected', updated_at = ? WHERE id = ?",
                (now_iso(), doc_id),
            )
            return

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM approval_steps WHERE document_id = ? AND decision = 'pending'",
            (doc_id,),
        )
        pending_count = cur.fetchone()["cnt"]
        if pending_count == 0:
            cur.execute(
                "UPDATE approval_documents SET status = 'approved', updated_at = ? WHERE id = ?",
                (now_iso(), doc_id),
            )
        else:
            cur.execute(
                "UPDATE approval_documents SET status = 'in_review', updated_at = ? WHERE id = ?",
                (now_iso(), doc_id),
            )

