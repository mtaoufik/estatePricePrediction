from deepseek_agent import DeepSeekAgent

def main():
    agent = DeepSeekAgent('models/deepseek_model.h5')
    input_data = [0.5, 2.2, 5.4, 2.1, 2.8, 1.5, 2.3, 1.9]  # Example input
    prediction = agent.predict(input_data)
    print(f'Prediction: {prediction}')

if __name__ == "__main__":
    main()