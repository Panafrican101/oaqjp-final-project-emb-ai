"""
Emotion detection application using the Watson NLP library.
"""

from __future__ import annotations
import re
from typing import Any, Dict, Optional

try:
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_watson.natural_language_understanding_v1 import EmotionOptions, Features
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    _WATSON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WATSON_AVAILABLE = False

DEFAULT_EMOTIONS = {
    "joy": 0.0,
    "sadness": 0.0,
    "anger": 0.0,
    "fear": 0.0,
    "disgust": 0.0,
}

KEYWORD_EMOTION_MAP = {
    "happy": "joy",
    "joy": "joy",
    "excited": "joy",
    "love": "joy",
    "good": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "terrible": "sadness",
    "angry": "anger",
    "hate": "anger",
    "mad": "anger",
    "fear": "fear",
    "scared": "fear",
    "worried": "fear",
    "disgust": "disgust",
    "gross": "disgust",
}


def _normalize_emotions(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values()) or 1.0
    return {key: round(value / total, 2) for key, value in scores.items()}


def _heuristic_emotions(text: str) -> Dict[str, float]:
    scores = DEFAULT_EMOTIONS.copy()
    words = re.findall(r"\w+", text.lower())
    for word in words:
        emotion = KEYWORD_EMOTION_MAP.get(word)
        if emotion:
            scores[emotion] += 1.0

    if not any(scores.values()):
        scores = {
            "joy": 0.2,
            "sadness": 0.2,
            "anger": 0.2,
            "fear": 0.2,
            "disgust": 0.2,
        }

    return _normalize_emotions(scores)


def emotion_detector(text: str, api_key: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
    """Detect emotion from text using Watson NLP if credentials are available.

    The function returns a standard response dictionary with status_code.
    """
    if not text or not text.strip():
        return {
            "input_text": text,
            "error": "Blank input provided",
            "status_code": 400,
        }

    cleaned_text = text.strip()
    emotions: Dict[str, float]
    source = "heuristic"

    if _WATSON_AVAILABLE and api_key and url:
        authenticator = IAMAuthenticator(api_key)
        nlu = NaturalLanguageUnderstandingV1(
            version="2021-08-01",
            authenticator=authenticator,
        )
        nlu.set_service_url(url)
        response = nlu.analyze(
            text=cleaned_text,
            features=Features(emotion=EmotionOptions()),
        ).get_result()
        emotions = response["emotion"]["document"]["emotion"]
        source = "watson"
    else:
        emotions = _heuristic_emotions(cleaned_text)

    top_emotion = max(emotions, key=emotions.get)
    return {
        "input_text": cleaned_text,
        "emotions": emotions,
        "top_emotion": top_emotion,
        "source": source,
        "status_code": 200,
    }


class EmotionDetectionApp:
    """Simple wrapper class for the emotion_detector application."""

    @staticmethod
    def analyze(text: str, api_key: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
        return emotion_detector(text, api_key=api_key, url=url)
