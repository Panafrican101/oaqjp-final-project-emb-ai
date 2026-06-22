import unittest

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):

    def test_emotion_detector_joy(self):
        result = emotion_detector("I am glad things are going well")
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_emotion_detector_anger(self):
        result = emotion_detector("I am really mad at you")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_emotion_detector_disgust(self):
        result = emotion_detector("I feel disgusted just hearing about it")
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_emotion_detector_sadness(self):
        result = emotion_detector("It is sad that this happened")
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_emotion_detector_fear(self):
        result = emotion_detector("I am scared to death")
        self.assertEqual(result["dominant_emotion"], "fear")

    def test_emotion_detector_blank_returns_none(self):
        result = emotion_detector("   ")
        self.assertIsNone(result["dominant_emotion"])
        self.assertIsNone(result["anger"])


if __name__ == "__main__":
    unittest.main()
