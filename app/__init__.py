from flask import Flask, jsonify, render_template

from app.db import get_connection, init_db, seed_defaults
from app.routes.approval import approval_bp
from app.routes.auth_routes import auth_bp
from app.routes.chatbot import chatbot_bp
from app.routes.org import org_bp


def create_app():
    app = Flask(__name__, template_folder="../templates")
    app.config["SECRET_KEY"] = "change-me-in-production"
    app.config["JSON_AS_ASCII"] = False

    init_db()
    seed_defaults()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.get("/ready")
    def ready():
        try:
            conn = get_connection()
            try:
                conn.execute("SELECT 1")
            finally:
                conn.close()
            return jsonify({"status": "ready", "db": "ok"}), 200
        except Exception as exc:
            return jsonify({"status": "not_ready", "db": "error", "reason": str(exc)}), 503

    @app.get("/build-id")
    def build_id():
        return jsonify({"build_id": "chatbot-2026-03-13-nextweek-fix-v1"}), 200

    app.register_blueprint(auth_bp)
    app.register_blueprint(chatbot_bp)
    app.register_blueprint(approval_bp)
    app.register_blueprint(org_bp)
    return app

