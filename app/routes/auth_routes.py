from flask import Blueprint, jsonify, request, session

from app.auth import get_current_user
from app.db import get_cursor, now_iso
from app.services.audit_service import log_action


auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.post("/login")
def login():
    data = request.json or {}
    employee_id = (data.get("employee_id") or "").strip()
    name = (data.get("name") or "").strip() or employee_id
    role = (data.get("role") or "user").strip()
    department = (data.get("department") or "").strip()
    position = (data.get("position") or "").strip()

    if not employee_id:
        return jsonify({"error": "employee_id는 필수입니다."}), 400
    if role not in ("user", "manager", "admin"):
        return jsonify({"error": "role은 user/manager/admin 중 하나여야 합니다."}), 400

    with get_cursor(commit=True) as cur:
        cur.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """
                INSERT INTO employees (employee_id, name, role, department, position, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (employee_id, name, role, department, position, now_iso()),
            )
        else:
            cur.execute(
                """
                UPDATE employees
                SET name = ?, role = ?, department = ?, position = ?
                WHERE employee_id = ?
                """,
                (
                    name or row["name"],
                    role or row["role"],
                    department or row["department"],
                    position or row["position"],
                    employee_id,
                ),
            )

    session["employee_id"] = employee_id
    session["name"] = name
    session["role"] = role
    log_action(employee_id, "auth.login", "employee", employee_id)
    return jsonify(status="success", user=get_current_user())


@auth_bp.post("/logout")
def logout():
    employee_id = session.get("employee_id")
    if employee_id:
        log_action(employee_id, "auth.logout", "employee", employee_id)
    session.clear()
    return jsonify(status="success")


@auth_bp.get("/me")
def me():
    user = get_current_user()
    if not user.get("employee_id"):
        return jsonify(logged_in=False, user=None)
    return jsonify(logged_in=True, user=user)

