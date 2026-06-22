"""
Emotion detection application using the Watson NLP library.
"""

import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_HEADERS = {
    "grpc-metadata-mm-model-id": (
        "emotion_aggregated-workflow_lang_en_stock"
    )
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

    input_json = {"raw_document": {"text": text_to_analyse}}
    response = requests.post(
        WATSON_URL,
        json=input_json,
        headers=WATSON_HEADERS,
        timeout=10,
    )

    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response_dict = response.json()
    emotions = (
        response_dict
        .get("emotionPredictions", [{}])[0]
        .get("emotion", {})
    )

    anger = emotions.get("anger", 0)
    disgust = emotions.get("disgust", 0)
    fear = emotions.get("fear", 0)
    joy = emotions.get("joy", 0)
    sadness = emotions.get("sadness", 0)

    scores = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
    }
    dominant_emotion = max(scores, key=scores.get)

    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion,
    }
