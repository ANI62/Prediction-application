import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_data(data):
    # Preprocess the data (e.g., encoding categorical features)
    data = pd.get_dummies(data)
    return data

def predict(data):
    # Preprocess the data
    data = preprocess_data(data)
    # Standardize the features
    data_scaled = scaler.transform(data)
    # Make predictions
    predictions = model.predict(data_scaled)
    return predictions
