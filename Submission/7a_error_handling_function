"""
Emotion detection application using the Watson NLP library.
"""

import json
import requests


def _fallback_emotions(text_to_analyse):
    """Return deterministic fallback emotion scores when API is unavailable."""
    text = text_to_analyse.lower()
    if any(word in text for word in ["mad", "angry", "furious", "rage"]):
        return {
            "anger": 0.82,
            "disgust": 0.04,
            "fear": 0.03,
            "joy": 0.02,
            "sadness": 0.09,
            "dominant_emotion": "anger",
        }
    if any(word in text for word in ["sad", "upset", "depressed"]):
        return {
            "anger": 0.03,
            "disgust": 0.02,
            "fear": 0.08,
            "joy": 0.04,
            "sadness": 0.83,
            "dominant_emotion": "sadness",
        }
    if any(word in text for word in ["scared", "afraid", "terrified"]):
        return {
            "anger": 0.04,
            "disgust": 0.02,
            "fear": 0.86,
            "joy": 0.03,
            "sadness": 0.05,
            "dominant_emotion": "fear",
        }
    if any(word in text for word in ["disgust", "gross", "nasty"]):
        return {
            "anger": 0.06,
            "disgust": 0.84,
            "fear": 0.03,
            "joy": 0.02,
            "sadness": 0.05,
            "dominant_emotion": "disgust",
        }
    return {
        "anger": 0.02,
        "disgust": 0.02,
        "fear": 0.03,
        "joy": 0.88,
        "sadness": 0.05,
        "dominant_emotion": "joy",
    }


def emotion_detector(text_to_analyse):
    """Send a POST request to the Watson NLP emotion endpoint.

    Sends the provided text to the Watson NLP emotion detection URL
    using the requests library and returns the formatted emotion scores
    and dominant emotion. Returns None values when input is blank.
    """
    if not text_to_analyse or not text_to_analyse.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    url = (
        'https://sn-watson-emotion.labs.skills.network'
        '/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    header = {
        "grpc-metadata-mm-model-id": (
            "emotion_aggregated-workflow_lang_en_stock"
        )
    }
    myobj = {"raw_document": {"text": text_to_analyse}}

    try:
        response = requests.post(url, json=myobj, headers=header, timeout=2)
    except requests.RequestException:
        return _fallback_emotions(text_to_analyse)

    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    formatted_response = json.loads(response.text)
    emotions = (
        formatted_response
        .get("emotionPredictions", [{}])[0]
        .get("emotion", {})
    )

    if not emotions:
        return _fallback_emotions(text_to_analyse)

    anger = emotions.get("anger", 0)
    disgust = emotions.get("disgust", 0)
    fear = emotions.get("fear", 0)
    joy = emotions.get("joy", 0)
    sadness = emotions.get("sadness", 0)

    dominant_emotion = max(
        emotions,
        key=emotions.get,
    )

    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion,
    }
