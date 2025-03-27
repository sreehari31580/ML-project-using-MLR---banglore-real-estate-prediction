import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
import os

def load_data():
    try:
        # Get base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_file = os.path.join(base_dir, 'realestate_data.csv')
        preprocessed_file = os.path.join(base_dir, 'preprocessed_realestate_data.csv')
        
        # If preprocessed file exists and is newer than raw data, use it
        if os.path.exists(preprocessed_file) and os.path.exists(data_file):
            if os.path.getmtime(preprocessed_file) > os.path.getmtime(data_file):
                print("Loading preprocessed data...")
                return pd.read_csv(preprocessed_file)
            
        # Load and preprocess raw data
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found at {data_file}")
            
        print("Processing raw data...")
        df = pd.read_csv(data_file)

        # Display initial data info
        print("Initial Data Info:")
        print(df.info())
        print("\nMissing Values:")
        print(df.isnull().sum())

        # Handle missing values more robustly
        numerical_cols = ['total_sqft', 'bath', 'balcony']
        for col in numerical_cols:
            if col in df.columns:
                # Fill missing values with median instead of forward fill
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)

        # Convert categorical variables to numerical
        categorical_columns = ['area_type', 'availability', 'location', 'size', 'society']
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        # Clean the 'total_sqft' column
        def convert_sqft(value):
            try:
                if isinstance(value, str):
                    if ' - ' in value:
                        parts = value.split(' - ')
                        if len(parts) == 2:
                            try:
                                low = float(re.sub(r'[^\d.]+', '', parts[0]))
                                high = float(re.sub(r'[^\d.]+', '', parts[1]))
                                return (low + high) / 2
                            except ValueError:
                                return None
                    cleaned_value = re.sub(r'[^\d.]+', '', value)
                    if cleaned_value:
                        return float(cleaned_value)
                elif isinstance(value, (int, float)):
                    return float(value)
                return None
            except Exception:
                return None

        # Apply conversion and handle invalid values
        df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
        
        # Replace invalid values with median
        median_sqft = df['total_sqft'].median()
        df['total_sqft'].fillna(median_sqft, inplace=True)

        # Remove outliers using IQR method
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        # Scale numerical features
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Save preprocessed data
        print(f"\nSaving preprocessed data to {preprocessed_file}")
        df.to_csv(preprocessed_file, index=False)

        return df

    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        raise
