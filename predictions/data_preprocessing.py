import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
import os

def clean_sqft(x):
    if isinstance(x, str):
        # Remove any text and keep only numbers
        nums = re.findall(r'\d+\.?\d*', x)
        if nums:
            # If range (e.g., "1000 - 1200"), take average
            if len(nums) > 1:
                return sum(float(num) for num in nums) / len(nums)
            return float(nums[0])
    return x

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

        # Clean total_sqft first
        df['total_sqft'] = df['total_sqft'].apply(clean_sqft)
        
        # Handle missing values
        numerical_cols = ['total_sqft', 'bath', 'balcony']
        for col in numerical_cols:
            if col in df.columns:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)

        # Convert categorical variables to numerical
        categorical_columns = ['area_type', 'availability', 'location', 'size', 'society']
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        # Convert price from lakhs to rupees (1 lakh = 100,000)
        df['price'] = df['price'] * 100000

        # Remove outliers using IQR method
        for col in numerical_cols + ['price']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # Save preprocessed data
        df.to_csv(preprocessed_file, index=False)
        print(f"Preprocessed data saved to {preprocessed_file}")
        print(f"\nPrice range: Rs. {df['price'].min():,.2f} to Rs. {df['price'].max():,.2f}")
        print(f"Average price: Rs. {df['price'].mean():,.2f}")
        return df
        
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        raise
