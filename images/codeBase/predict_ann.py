"""
ANN Model Prediction Script for Docker Container
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def load_and_predict():
    print("\n" + "="*70)
    print("ANN MODEL PREDICTION")
    print("="*70 + "\n")
    
    model_path = '/shared_data/knowledgeBase/currentAiSolution.h5'
    activation_path = '/shared_data/activationBase/activation_data.csv'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(activation_path):
        print(f"Error: Activation data not found at {activation_path}")
        return
    
    try:
        print("Loading ANN model...")
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!\n")
        
        print("Loading activation data...")
        df = pd.read_csv(activation_path)
        print(f"Loaded {len(df)} sample(s)\n")
        
        columns_to_remove = ['Unnamed: 0', 'Risk', 'Risk_encoded', 'Sex', 'Housing', 
                            'Saving accounts', 'Checking account', 'Purpose']
        
        X = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        X = X.select_dtypes(include=[np.number])
        
        print(f"Features used: {list(X.columns)}")
        print(f"Number of features: {X.shape[1]}\n")
        
        print("Making predictions...")
        predictions_proba = model.predict(X, verbose=0)
        predictions = (predictions_proba > 0.5).astype(int).flatten()
        
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70 + "\n")
        
        for i, (pred, prob) in enumerate(zip(predictions, predictions_proba)):
            risk_label = "GOOD CREDIT" if pred == 1 else "❌ BAD CREDIT"
            confidence = prob[0] if pred == 1 else (1 - prob[0])
            
            print(f"Sample {i+1}:")
            print(f"  Prediction: {risk_label}")
            print(f"  Probability: {prob[0]:.4f}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Classification: {'Good' if pred == 1 else 'Bad'}")
            print()
        
        print("="*70)
        print("ANN Prediction Complete!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_and_predict()