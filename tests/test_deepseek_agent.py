import unittest
from deepseek_agent import DeepSeekAgent

class TestDeepSeekAgent(unittest.TestCase):
    def test_model_loading(self):
        agent = DeepSeekAgent('models/deepseek_model.h5')
        self.assertIsNotNone(agent.model)

    def test_prediction(self):
        agent = DeepSeekAgent('models/deepseek_model.h5')
        input_data = [0.5, 1.2, 3.4, 2.1, 0.8, 1.5, 2.3, 0.9]
        prediction = agent.predict(input_data)
        self.assertEqual(len(prediction), 1)

if __name__ == '__main__':
    unittest.main()