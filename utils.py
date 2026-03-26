import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)


def preprocess_data(df):
    """Preprocess the data (e.g., handle missing values, encoding)."""
    # Example: fill missing values
    df.fillna(method='ffill', inplace=True)
    return df


def normalize_data(df):
    """Normalize the data using standard scaling."""
    scaler = StandardScaler()
    return scaler.fit_transform(df), scaler


def load_model(model_path):
    """Load a machine learning model from a file."""
    return joblib.load(model_path)


def make_predictions(model, data):
    """Make predictions using the model."""
    return model.predict(data)


def convert_price_category(prediction):
    """Convert numerical prediction to price category."""
    if prediction < 100:
        return 'Low'
    elif 100 <= prediction < 300:
        return 'Medium'
    else:
        return 'High'


def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    return mse
