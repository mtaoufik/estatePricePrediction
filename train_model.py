from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os
import pickle

def create_and_train_model():
    # Load California Housing dataset
    california = fetch_california_housing()
    X, y = california.data, california.target
    feature_names = california.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test
    _split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for future use
    with open('data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Create and compile the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Save the trained model
    model.save('models/deepseek_model.h5')

    print('Model trained and saved successfully.')