from flask import Blueprint, jsonify, request

from app.auth import login_required, roles_required
from app.db import get_cursor
from app.services.audit_service import log_action


org_bp = Blueprint("org", __name__, url_prefix="/org")


@org_bp.get("/employees")
@login_required
def list_employees():
    with get_cursor() as cur:
        cur.execute("SELECT * FROM employees ORDER BY employee_id")
        rows = cur.fetchall()
    return jsonify(rows=rows)


@org_bp.post("/departments")
@roles_required("manager", "admin")
def add_department():
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "name은 필수입니다."}), 400
    with get_cursor(commit=True) as cur:
        cur.execute("INSERT OR IGNORE INTO departments (name) VALUES (?)", (name,))
    log_action("system", "org.add_department", "department", payload={"name": name})
    return jsonify(status="success", name=name)


@org_bp.get("/departments")
@login_required
def list_departments():
    with get_cursor() as cur:
        cur.execute("SELECT * FROM departments ORDER BY name")
        rows = cur.fetchall()
    return jsonify(rows=rows)


@org_bp.post("/positions")
@roles_required("manager", "admin")
def add_position():
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "name은 필수입니다."}), 400
    with get_cursor(commit=True) as cur:
        cur.execute("INSERT OR IGNORE INTO positions (name) VALUES (?)", (name,))
    log_action("system", "org.add_position", "position", payload={"name": name})
    return jsonify(status="success", name=name)


@org_bp.get("/positions")
@login_required
def list_positions():
    with get_cursor() as cur:
        cur.execute("SELECT * FROM positions ORDER BY name")
        rows = cur.fetchall()
    return jsonify(rows=rows)

