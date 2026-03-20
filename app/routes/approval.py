from flask import Blueprint, jsonify, request, session

from app.auth import login_required, roles_required
from app.db import get_cursor
from app.services.approval_service import create_document, decide_document, submit_document
from app.services.audit_service import log_action


approval_bp = Blueprint("approval", __name__, url_prefix="/approval")


@approval_bp.post("/documents")
@login_required
def create_approval_document():
    data = request.json or {}
    doc_type = (data.get("doc_type") or "draft").strip()
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    approvers = data.get("approvers") or []
    source_query = (data.get("source_query") or "").strip()
    source_answer = (data.get("source_answer") or "").strip()

    if not title or not body:
        return jsonify({"error": "title/body는 필수입니다."}), 400

    drafter = session.get("employee_id")
    doc_id = create_document(
        doc_type=doc_type,
        title=title,
        body=body,
        drafter_employee_id=drafter,
        approvers=approvers,
        source_query=source_query,
        source_answer=source_answer,
    )
    log_action(drafter, "approval.create", "approval_document", doc_id, {"doc_type": doc_type})
    return jsonify(status="success", document_id=doc_id)


@approval_bp.post("/documents/<int:doc_id>/submit")
@login_required
def submit_approval_document(doc_id):
    submit_document(doc_id)
    log_action(session.get("employee_id"), "approval.submit", "approval_document", doc_id)
    return jsonify(status="success")


@approval_bp.post("/documents/<int:doc_id>/decision")
@roles_required("manager", "admin")
def decision_approval_document(doc_id):
    data = request.json or {}
    decision = (data.get("decision") or "").strip()
    comment = (data.get("comment") or "").strip()
    approver = session.get("employee_id")
    try:
        decide_document(doc_id, approver, decision, comment)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    log_action(approver, "approval.decision", "approval_document", doc_id, {"decision": decision})
    return jsonify(status="success")


@approval_bp.get("/documents")
@login_required
def list_documents():
    employee_id = session.get("employee_id")
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT d.*
            FROM approval_documents d
            LEFT JOIN approval_steps s ON s.document_id = d.id
            WHERE d.drafter_employee_id = ? OR s.approver_employee_id = ?
            ORDER BY d.id DESC
            """,
            (employee_id, employee_id),
        )
        docs = cur.fetchall()
    return jsonify(rows=docs)


@approval_bp.get("/documents/<int:doc_id>")
@login_required
def get_document(doc_id):
    with get_cursor() as cur:
        cur.execute("SELECT * FROM approval_documents WHERE id = ?", (doc_id,))
        doc = cur.fetchone()
        if not doc:
            return jsonify({"error": "문서를 찾을 수 없습니다."}), 404
        cur.execute(
            "SELECT * FROM approval_steps WHERE document_id = ? ORDER BY step_order",
            (doc_id,),
        )
        steps = cur.fetchall()
    return jsonify(document=doc, steps=steps)


@approval_bp.post("/from-chatbot")
@login_required
def create_from_chatbot():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    answer = (data.get("answer") or "").strip()
    label = (data.get("label") or "").strip() or "기타"
    approvers = data.get("approvers") or []

    if not query:
        return jsonify({"error": "query는 필수입니다."}), 400

    title = f"[자동초안/{label}] {query[:40]}"
    body = f"질문:\n{query}\n\nAI 초안:\n{answer or '(답변 없음)'}"
    doc_id = create_document(
        doc_type="chatbot_draft",
        title=title,
        body=body,
        drafter_employee_id=session.get("employee_id"),
        approvers=approvers,
        source_query=query,
        source_answer=answer,
    )
    log_action(
        session.get("employee_id"),
        "approval.create_from_chatbot",
        "approval_document",
        doc_id,
        {"label": label},
    )
    return jsonify(status="success", document_id=doc_id, title=title)

