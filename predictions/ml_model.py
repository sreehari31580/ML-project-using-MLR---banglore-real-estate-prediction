import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from .data_preprocessing import load_data
import os

# Define base directory for model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'realestate_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'target_scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'preprocessed_realestate_data.csv')

def train_model():
    try:
        # Load the preprocessed dataset
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            raise FileNotFoundError(f"Preprocessed data file not found at {DATA_PATH}")

        # Print price range for debugging
        print(f"Price range in dataset: ₹{df['price'].min():,.2f} to ₹{df['price'].max():,.2f}")
        print(f"Average price: ₹{df['price'].mean():,.2f}")

        # Define features and target variable
        X = df.drop('price', axis=1)  # Features
        y = df['price']  # Target variable

        # Initialize scaler for target variable only
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'Mean Squared Error: {mse}')

        # Save the model and scaler
        dump(model, MODEL_PATH)
        dump(target_scaler, SCALER_PATH)
        print("Model and scaler saved successfully")

        return model
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return load(MODEL_PATH)

def load_scalers():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
    return load(SCALER_PATH)
