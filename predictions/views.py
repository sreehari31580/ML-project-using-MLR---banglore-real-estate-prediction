from django.shortcuts import render
from .ml_model import load_model, load_scalers
import numpy as np
import pandas as pd
import os

def home(request):
    return render(request, 'predictions/home.html')

def predict(request):
    if request.method == 'POST':
        try:
            # Input validation
            total_sqft = float(request.POST['total_sqft'])
            bath = int(request.POST['bath'])
            balcony = int(request.POST['balcony'])

            # Validate ranges
            if total_sqft <= 0:
                raise ValueError("Total square feet must be greater than 0")
            if bath <= 0:
                raise ValueError("Number of bathrooms must be greater than 0")
            if balcony < 0:
                raise ValueError("Number of balconies cannot be negative")
            if total_sqft > 10000:  # Reasonable upper limit
                raise ValueError("Total square feet seems unusually large")
            if bath > 10:  # Reasonable upper limit
                raise ValueError("Number of bathrooms seems unusually large")
            if balcony > 10:  # Reasonable upper limit
                raise ValueError("Number of balconies seems unusually large")

            # Load model and scaler
            try:
                model = load_model()
                target_scaler = load_scalers()
            except FileNotFoundError:
                return render(request, 'predictions/predict.html', {
                    'error': "The prediction model has not been trained yet. Please contact the administrator."
                })
            except Exception as e:
                return render(request, 'predictions/predict.html', {
                    'error': "Error loading the prediction model. Please try again later."
                })

            # Create initial DataFrame with numerical features
            input_data = pd.DataFrame({
                'total_sqft': [total_sqft],
                'bath': [bath],
                'balcony': [balcony]
            })

            # Get all expected features from the model
            expected_features = model.feature_names_in_

            # Initialize all expected features with 0
            for feature in expected_features:
                if feature not in input_data.columns:
                    input_data[feature] = 0

            # Ensure columns are in the same order as training
            input_data = input_data[expected_features]

            # Make prediction
            prediction_scaled = model.predict(input_data)

            # Inverse transform the prediction to get actual price in lakhs
            prediction_lakhs = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]

            # Convert lakhs to rupees (1 lakh = 100,000 rupees)
            prediction_rupees = prediction_lakhs * 100000

            # Format the prediction to 2 decimal places
            formatted_prediction = "{:,.2f}".format(prediction_rupees)

            return render(request, 'predictions/result.html', {
                'prediction': formatted_prediction,
                'original_sqft': total_sqft,
                'bath': bath,
                'balcony': balcony
            })
        
        except ValueError as e:
            return render(request, 'predictions/predict.html', {'error': str(e)})
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render(request, 'predictions/predict.html', {
                'error': "An unexpected error occurred. Please check your input and try again."
            })

    return render(request, 'predictions/predict.html')
