"""Flask deployment for the emotion detection application."""

# pylint: disable=consider-using-f-string

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector


APP = Flask("Emotion Detector")


@APP.route("/emotionDetector")
def sent_analyzer() -> str:
    """Analyze emotion for the submitted text query parameter."""
    text_to_analyse = request.args.get("textToAnalyze", "")
    response = emotion_detector(text_to_analyse)

    if response["dominant_emotion"] is None:
        return "Invalid input! Try again."

    return (
        "For the given statement, the system response is "
        "'anger': {}, 'disgust': {}, 'fear': {}, "
        "'joy': {} and 'sadness': {}. "
        "The dominant emotion is {}."
    ).format(
            response["anger"],
            response["disgust"],
            response["fear"],
            response["joy"],
            response["sadness"],
            response["dominant_emotion"],
    )


@APP.route("/")
def render_index_page() -> str:
    """Serve the application home page."""
    return render_template("index.html")


@APP.route("/analyze")
def render_index_page_alias() -> str:
    """Serve the application home page for legacy path compatibility."""
    return render_template("index.html")


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
