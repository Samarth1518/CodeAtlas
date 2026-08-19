"""
CodeAtlas - AI-Powered Codebase Intelligence Platform
Backend Entry Point
"""

import os

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

from routes.repos import repos_bp
from routes.contents import contents_bp
from routes.chunks import chunks_bp
from routes.index import index_bp
from routes.search import search_bp
from routes.chat import chat_bp
from routes.insights import insights_bp

# Load .env into os.environ as early as possible so all downstream
# code (blueprints, services) can read variables via os.getenv().
# override=False means real environment variables always win over .env.
load_dotenv(override=False)


def create_app():
    app = Flask(__name__)
    CORS(app)

    # ── Blueprints ──────────────────────────────────────────────────────────
    app.register_blueprint(repos_bp, url_prefix="/api/repos")
    app.register_blueprint(contents_bp, url_prefix="/api/contents")
    app.register_blueprint(chunks_bp, url_prefix="/api/chunks")
    app.register_blueprint(index_bp, url_prefix="/api/index")
    app.register_blueprint(search_bp, url_prefix="/api/search")
    app.register_blueprint(chat_bp, url_prefix="/api/chat")
    app.register_blueprint(insights_bp, url_prefix="/api/insights")

    # ── Core routes ─────────────────────────────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "ok", "service": "CodeAtlas API"})

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") in ("1", "true", "True") and os.getenv("FLASK_ENV") != "production"
    app = create_app()
    app.run(debug=debug, host="0.0.0.0", port=port)
