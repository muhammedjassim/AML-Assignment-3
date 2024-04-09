import unittest
import joblib
from score import score
import requests
import subprocess
import time
import json
from app import app


class TestScore(unittest.TestCase):
    def setUp(self):
        # Load the trained model
        self.model = joblib.load('./trained_model.joblib')

    def test_score(self):
        # Smoke test
        prediction, propensity = score("This is a sample text.", self.model, 0.5)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(propensity)

        # Format test
        self.assertIsInstance(prediction, int)
        self.assertIsInstance(propensity, float)

        # Prediction and propensity value tests
        self.assertIn(prediction, [0, 1])
        self.assertTrue(0.0 <= propensity <= 1.0)

        # Threshold tests
        prediction, propensity = score("This is a sample text.", self.model, 0.0)
        self.assertEqual(prediction, 1)
        prediction, propensity = score("This is a sample text.", self.model, 1.0)
        self.assertEqual(prediction, 0)

        # Obvious spam and non-spam tests
        spam_text = "Buy this product now! Act fast!"
        non_spam_text = "This is a normal message."
        spam_prediction, spam_propensity = score(spam_text, self.model, 0.5)
        non_spam_prediction, non_spam_propensity = score(non_spam_text, self.model, 0.5)
        self.assertEqual(spam_prediction, 1)
        self.assertEqual(non_spam_prediction, 0)


class TestFlask(unittest.TestCase):
    def setUp(self):
        self.proc = None

    def tearDown(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()

    def test_flask(self):
        # Launch the Flask app in a subprocess
        self.proc = subprocess.Popen(['flask', 'run', '--port=5000'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for the server to start
        time.sleep(2)

        try:
            # Test the /score endpoint
            response = requests.post('http://localhost:5000/score', data={'text': 'This is a test text.'}, timeout=10)
            self.assertEqual(response.status_code, 200)

            data = json.loads(response.text)
            self.assertIsInstance(data['prediction'], int)
            self.assertIsInstance(data['propensity'], float)
            self.assertTrue(0.0 <= data['propensity'] <= 1.0)
        finally:
            # Terminate the Flask app
            self.proc.terminate()
            self.proc.wait()


if __name__ == '__main__':
    unittest.main()