"""Flask deployment for the emotion detection application."""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request

from EmotionDetection.emotion_detection import emotion_detector


APP = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Emotion Detection</title>
</head>
<body>
  <h1>Emotion Detection</h1>
  <form action="/emotionDetector" method="post">
    <label for="text">Enter text:</label><br />
    <textarea id="text" name="text" rows="4" cols="50">
      {{ text or '' }}
    </textarea>
    <br />
    <button type="submit">Analyze</button>
  </form>
  {% if error %}
    <p style="color: red; font-weight: bold;">{{ error }}</p>
  {% endif %}
  {% if result %}
    <h2>Analysis Result</h2>
    <p>Top emotion: {{ result.top_emotion }}</p>
    <ul>
      {% for emotion, score in result.emotions.items() %}
        <li>{{ emotion }}: {{ score }}</li>
      {% endfor %}
    </ul>
  {% endif %}
</body>
</html>
"""


@APP.route("/")
def index() -> str:
    """Render the emotion detection form."""
    return render_template_string(INDEX_HTML, error=None, result=None, text="")


@APP.route("/emotionDetector", methods=["POST"])
def emotion_detector_route() -> Any:
    """Handle emotion detection requests and return JSON or page output."""
    text = request.form.get("text", "")
    if not text or not text.strip():
        return render_template_string(
            INDEX_HTML,
            error="Invalid input! Try again.",
            result=None,
            text=text,
        )

    result = emotion_detector(text)
    status = result.get("status_code", 200)
    return (
        render_template_string(
            INDEX_HTML,
            error=None,
            result=result,
            text=text,
        ),
        status,
    )


@APP.route("/emotionDetector/json", methods=["POST"])
def emotion_detector_json_route() -> Any:
    """Return a JSON response for emotion detection requests."""
    payload = request.get_json(silent=True) or {}
    text = (
        payload.get("text")
        if payload
        else request.form.get("text", "")
    )
    if not text or not text.strip():
        return (
            jsonify({"error": "Blank input provided", "status_code": 400}),
            400,
        )

    result = emotion_detector(text)
    return jsonify(result), result.get("status_code", 200)


def run_static_analysis() -> str:
    """Run static analysis for the server module."""
    root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-m",
        "flake8",
        str(root / "server.py"),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Static analysis failed:\n"
            f"{completed.stdout}\n"
            f"{completed.stderr}"
        )
    return completed.stdout.strip()


if __name__ == "__main__":
    if "--lint" in sys.argv:
        message = run_static_analysis()
        print(message or "Static analysis passed with no issues.")
    else:
        APP.run(host="127.0.0.1", port=5000, debug=False)
