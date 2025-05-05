import pickle
from sklearn.preprocessing import StandardScaler

# Example: Save scaler
scaler = StandardScaler()
with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved to 'data/scaler.pkl'.")