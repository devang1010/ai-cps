"""
OLS Model Prediction Script for Docker Container
"""

import pandas as pd
import numpy as np
import pickle
import os

def load_and_predict():
    print("\n" + "="*70)
    print("OLS MODEL PREDICTION")
    print("="*70 + "\n")
    
    model_path = '/shared_data/knowledgeBase/currentOlsSolution.pkl'
    activation_path = '/shared_data/activationBase/activation_data.csv'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(activation_path):
        print(f"Error: Activation data not found at {activation_path}")
        return
    
    try:
        print("Loading OLS model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!\n")
        
        print("Loading activation data...")
        df = pd.read_csv(activation_path)
        print(f"Loaded {len(df)} sample(s)\n")
        
        columns_to_remove = ['Unnamed: 0', 'Risk', 'Risk_encoded', 'Sex', 'Housing', 
                            'Saving accounts', 'Checking account', 'Purpose']
        
        X = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        X = X.select_dtypes(include=[np.number])
        
        print(f"Features before adding constant: {list(X.columns)}")
        print(f"Number of features: {X.shape[1]}")
        
        # Add constant for OLS
        import statsmodels.api as sm
        X_const = sm.add_constant(X, has_constant='add')
        
        print(f"Number of features after adding constant: {X_const.shape[1]}")
        print(f"Model expects {len(model.params)} parameters\n")
        
        # Check if dimensions match
        if X_const.shape[1] != len(model.params):
            print(f"Warning: Feature mismatch!")
            print(f"   Activation data has {X_const.shape[1]} features")
            print(f"   Model expects {len(model.params)} features")
            print(f"\n Model parameter names:")
            print(f"   {list(model.params.index)}")
            print(f"\n Available feature names:")
            print(f"   {list(X_const.columns)}")
            print("\n Cannot proceed with prediction due to feature mismatch.")
            return
        
        print("Making predictions...")
        predictions_proba = model.predict(X_const)
        predictions = (predictions_proba > 0.5).astype(int)
        
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70 + "\n")
        
        for i, (pred, prob) in enumerate(zip(predictions, predictions_proba)):
            risk_label = " GOOD CREDIT" if pred == 1 else "BAD CREDIT"
            confidence = prob if pred == 1 else (1 - prob)
            
            print(f"Sample {i+1}:")
            print(f"  Prediction: {risk_label}")
            print(f"  Probability: {prob:.4f}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Classification: {'Good' if pred == 1 else 'Bad'}")
            print()
        
        print("="*70)
        print("OLS Prediction Complete!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_and_predict()