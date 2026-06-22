import unittest

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):

    def test_import_emotion_detector(self):
        self.assertTrue(callable(emotion_detector))

    def test_emotion_detector_returns_expected_format(self):
        result = emotion_detector("I am happy and excited")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status_code"], 200)
        self.assertIn("emotions", result)
        self.assertIn("top_emotion", result)
        self.assertEqual(result["top_emotion"], "joy")
        self.assertIsInstance(result["emotions"], dict)
        self.assertIn("joy", result["emotions"])

    def test_emotion_detector_blank_input_returns_400(self):
        result = emotion_detector("   ")
        self.assertEqual(result["status_code"], 400)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Blank input provided")

    def test_emotion_detector_top_emotion_calculation(self):
        result = emotion_detector("I feel sad and terrible")
        self.assertEqual(result["top_emotion"], "sadness")
        self.assertGreater(result["emotions"]["sadness"], 0)


if __name__ == "__main__":
    unittest.main()
