import pickle

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)