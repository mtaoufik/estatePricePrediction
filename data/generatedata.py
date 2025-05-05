from sklearn.datasets import fetch_california_housing
import pandas as pd

# Fetch the dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['Target'] = california.target

# Save to CSV
data.to_csv('data/california_housing_data.csv', index=False)
print("Dataset saved to 'data/california_housing_data.csv'.")