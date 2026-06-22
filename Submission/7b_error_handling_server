"""Flask deployment for the emotion detection application."""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request

from EmotionDetection import emotion_detector


APP = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Emotion Detection</title>
</head>
<body>
  <h1>Emotion Detection</h1>
  <form action="/analyze" method="post">
    <label for="text">Enter text:</label>
    <textarea id="text" name="text" rows="4" cols="50"></textarea>
    <br />
    <button type="submit">Analyze</button>
  </form>
</body>
</html>
"""


@APP.route("/")
def index() -> str:
    return render_template_string(INDEX_HTML)


@APP.route("/analyze", methods=["POST"])
def analyze() -> Any:
    text = request.form.get("text", "")
    if not text or not text.strip():
        return (
            jsonify({"error": "Blank input provided", "status_code": 400}),
            400,
        )

    result = emotion_detector(text)
    status = result.get("status_code", 200)
    return jsonify(result), status


def run_static_analysis() -> str:
    """Run static analysis for the server module."""
    root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-m",
        "flake8",
        str(root / "server.py"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Static analysis failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout.strip()


if __name__ == "__main__":
    if "--lint" in sys.argv:
        message = run_static_analysis()
        print(message or "Static analysis passed with no issues.")
    else:
        APP.run(host="127.0.0.1", port=5000, debug=False)
