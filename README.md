# Estate Price Prediction

## Objective
The **Estate Price Prediction** project is designed to develop a machine learning model that predicts real estate property prices based on various features such as location, size, and other relevant factors. This repository aims to provide tools for data preprocessing, model training, and prediction, with a focus on ease of use and scalability.

## Features
- **Data Processing**: Tools to clean and preprocess real estate data.
- **Model Training**: Scripts to train models using advanced machine learning techniques.
- **Prediction**: Easily make predictions using trained models.
- **Testing**: Unit tests to ensure code reliability.

## Structure
```
estatePricePredict/
│
├── deepseek_agent.py        # Core module for prediction logic
├── train_model.py           # Script for training the model
├── predict_model.py         # Script for making predictions
├── utils.py                 # Utility functions for data handling
│
├── tests/                   # Unit tests
│   ├── test_deepseek_agent.py
│
├── data/                    # Sample datasets and preprocessing artifacts
│   ├── california_housing_data.csv
│
├── models/                  # Trained models saved in this directory
│   ├── deepseek_model.h5
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Requirements
- Python 3.9 or higher
- Libraries: TensorFlow, scikit-learn, numpy, pandas

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/mtaoufik/estatePricePrediction.git
   cd estatePricePrediction

--------------------------------------------------------------
Install dependencies:
pip install -r requirements.txt

Train the model:
python train_model.py

Make predictions:
python predict_model.py
