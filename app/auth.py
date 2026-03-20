from functools import wraps
from flask import session, jsonify


def get_current_user():
    return {
        "employee_id": session.get("employee_id"),
        "name": session.get("name"),
        "role": session.get("role"),
    }


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("employee_id"):
            return jsonify({"error": "로그인이 필요합니다."}), 401
        return fn(*args, **kwargs)

    return wrapper


def roles_required(*roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not session.get("employee_id"):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            if session.get("role") not in roles:
                return jsonify({"error": "권한이 없습니다."}), 403
            return fn(*args, **kwargs)

        return wrapper

    return decorator

