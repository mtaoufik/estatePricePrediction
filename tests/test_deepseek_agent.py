import unittest
import sys
import os

# Adjust Python path to include the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the DeepSeekAgent class
from deepseek_agent import DeepSeekAgent

class TestDeepSeekAgent(unittest.TestCase):
    def test_model_loading(self):
        model_path = os.path.join(os.path.dirname(__file__), '../models/deepseek_model.h5')
        agent = DeepSeekAgent(model_path)
        self.assertIsNotNone(agent.model)

    def test_prediction(self):
        model_path = os.path.join(os.path.dirname(__file__), '../models/deepseek_model.h5')
        agent = DeepSeekAgent(model_path)
        input_data = [1.5, 1.9, 5.4, 5.1, 5.8, 5.5, 5.3, 3.9]
        prediction = agent.predict(input_data)
        self.assertEqual(len(prediction), 1)

if __name__ == '__main__':
    unittest.main()
