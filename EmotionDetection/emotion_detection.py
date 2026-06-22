"""
Emotion detection application using the Watson NLP library.
"""

import json
import requests


def emotion_detector(text_to_analyse):
    """Send a POST request to the Watson NLP emotion endpoint.

    Sends the provided text to the Watson NLP emotion detection URL
    using the requests library and returns the formatted emotion scores
    and dominant emotion. Returns None values when input is blank.
    """
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

    response = requests.post(url, json=myobj, headers=header, timeout=10)

    if response.status_code == 400 or not text_to_analyse.strip():
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
