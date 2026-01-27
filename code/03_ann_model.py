"""
ANN Model Training for German Credit Risk Classification
Author: Devang Thaker & Krish Manvar
Date: January 2026
Course: M. Grum: Advanced AI-based Application Systems

This script:
1. Loads and prepares the German Credit Risk data
2. Builds an Artificial Neural Network (ANN) using TensorFlow/Keras
3. Trains the model with training data
4. Validates with test data
5. Generates all required visualizations:
   - Training/testing curves
   - Diagnostic plots
   - Scatter plots
6. Saves the trained model and all performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class GermanCreditANN:
    """
    Artificial Neural Network for German Credit Risk Classification
    """
    
    def __init__(self, output_dir='learningBase'):
        """
        Initialize the ANN model
        
        Args:
            output_dir (str): Directory to save all outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.training_metrics = {}
        
        print("=" * 70)
        print("🧠 German Credit Risk ANN Model")
        print("=" * 70)
        print()
    
    
    def load_data(self, train_path='data/training_data.csv', test_path='data/test_data.csv'):
        """
        Load and prepare training and test data
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data
        """
        print("📂 Loading data...")
        
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # --------------------------------------------------
        # Remove leakage and index columns
        # --------------------------------------------------
        leakage_cols = [
            'Risk_encoded',      # target leakage
            'Unnamed: 0'         # index column
        ]

        train_df = train_df.drop(columns=[c for c in leakage_cols if c in train_df.columns])
        test_df  = test_df.drop(columns=[c for c in leakage_cols if c in test_df.columns])
        
        print(f"   Training samples: {len(train_df)}")
        print(f"   Test samples: {len(test_df)}")
        print()
        
        # --------------------------------------------------
        # Explicit target column
        # --------------------------------------------------
        target_col = 'Risk'
        print(f"🎯 Target column (explicitly set): {target_col}")
        print()
        
        # Separate features and target
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Select only numeric features
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test.select_dtypes(include=[np.number])
        
        print(f"📊 Features: {X_train.shape[1]}")
        print(f"   Feature names: {list(X_train.columns)}")
        print()
        
        # --------------------------------------------------
        # Encode target explicitly: bad=0, good=1
        # --------------------------------------------------
        print("🔤 Encoding target variable (Risk)...")

        risk_mapping = {'bad': 0, 'good': 1}
        y_train = y_train.map(risk_mapping).astype(int)
        y_test  = y_test.map(risk_mapping).astype(int)          
        print("   Target mapping: bad → 0, good → 1")
        print(f"   Class distribution (train): {np.bincount(y_train)}")
        print(f"   Class distribution (test): {np.bincount(y_test)}")
        print()

        
        # Convert to binary if needed (0 and 1)
        y_train = np.where(y_train > 0, 1, 0)
        y_test = np.where(y_test > 0, 1, 0)
        
        print(f"   Class distribution (train): {np.bincount(y_train)}")
        print(f"   Class distribution (test): {np.bincount(y_test)}")
        print()
        
        # Standardize features
        print("📏 Standardizing features...")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print("✅ Data loaded and prepared successfully!")
        print()
    
    
    def build_model(self):
        """
        Build the ANN architecture
        """
        print("🏗️  Building ANN model...")
        
        input_dim = self.X_train.shape[1]
        
        # Create Sequential model
        self.model = Sequential([
            Input(shape=(input_dim,), name='input_layer'),
            # Input layer
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layer 1
            Dense(64, activation='relu', name='hidden_layer_1'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layer 2
            Dense(32, activation='relu', name='hidden_layer_2'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Hidden layer 3
            Dense(16, activation='relu', name='hidden_layer_3'),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print("\n📋 Model Architecture:")
        print("-" * 70)
        self.model.summary()
        print("-" * 70)
        print()
    
    
    def train_model(self, epochs=100, batch_size=32):
        """
        Train the ANN model
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        print(f"🚀 Training model for {epochs} epochs...")
        print()
        
        # Define callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint to save best model
            ModelCheckpoint(
                os.path.join(self.output_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when learning plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print()
        print("✅ Training completed!")
        print()
    
    
    def evaluate_model(self):
        """
        Evaluate the model and generate metrics
        """
        print("📊 Evaluating model...")
        print()
        
        # Make predictions
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # Get training history
        final_epoch = len(self.history.history['loss'])
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        
        # Store metrics
        self.training_metrics = {
            'total_epochs_trained': final_epoch,
            'final_training_loss': float(final_train_loss),
            'final_validation_loss': float(final_val_loss),
            'final_training_accuracy': float(final_train_acc),
            'final_validation_accuracy': float(final_val_acc),
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1),
            'total_training_samples': int(len(self.X_train)),
            'total_test_samples': int(len(self.X_test)),
            'number_of_features': int(self.X_train.shape[1])
        }
        
        # Print results
        print("=" * 70)
        print("📈 TRAINING RESULTS")
        print("=" * 70)
        print(f"\n🔄 Training Information:")
        print(f"   Total Epochs: {final_epoch}")
        print(f"   Training Samples: {len(self.X_train)}")
        print(f"   Test Samples: {len(self.X_test)}")
        print(f"   Number of Features: {self.X_train.shape[1]}")
        
        print(f"\n📉 Final Loss Values:")
        print(f"   Training Loss: {final_train_loss:.4f}")
        print(f"   Validation Loss: {final_val_loss:.4f}")
        
        print(f"\n🎯 Final Accuracy Values:")
        print(f"   Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"   Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        
        print(f"\n✅ Test Set Performance:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print("=" * 70)
        print()
        
        # Classification report
        print("📋 Detailed Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Bad Credit', 'Good Credit']))
        print()
        
        return y_pred, y_pred_proba
    
    
    def save_model(self):
        """
        Save the trained ANN model (assignment-compliant)
        """
        print("💾 Saving trained model...")

        model_path = os.path.join(self.output_dir, 'currentAiSolution.h5')
        self.model.save(model_path)

        print(f"   ✓ Saved trained ANN model: {model_path}")
        print()

    
    
    def save_metrics(self):
        """
        Save training metrics and performance data
        """
        print("📊 Saving training metrics...")
        
        # Save metrics as JSON
        metrics_path = os.path.join(self.output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=4)
        print(f"   ✓ Saved metrics: {metrics_path}")
        
        # Save training history
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
        }
        
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        print(f"   ✓ Saved training history: {history_path}")
        
        # Save as CSV for easy analysis
        history_df = pd.DataFrame(self.history.history)
        history_csv_path = os.path.join(self.output_dir, 'training_history.csv')
        history_df.to_csv(history_csv_path, index=False)
        print(f"   ✓ Saved training history CSV: {history_csv_path}")
        
        print()
    
    
    def plot_training_curves(self):
        """
        Plot training and testing curves (Loss and Accuracy)
        """
        print("📈 Generating training/testing curves...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_path = os.path.join(self.output_dir, 'training_testing_curves.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved training curves: {curve_path}")
        plt.close()
    
    
    def plot_diagnostic_plots(self, y_pred, y_pred_proba):
        """
        Generate diagnostic plots (Confusion Matrix, ROC Curve, Precision-Recall Curve)
        
        Args:
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
        """
        print("🔬 Generating diagnostic plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Bad Credit', 'Good Credit'],
                   yticklabels=['Bad Credit', 'Good Credit'])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label', fontsize=12)
        axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
        axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right", fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        
        axes[1, 0].plot(recall, precision, color='blue', lw=2)
        axes[1, 0].set_xlabel('Recall', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        axes[1, 1].hist(y_pred_proba[self.y_test == 0], bins=30, alpha=0.6, label='Bad Credit', color='red')
        axes[1, 1].hist(y_pred_proba[self.y_test == 1], bins=30, alpha=0.6, label='Good Credit', color='green')
        axes[1, 1].set_xlabel('Predicted Probability', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        diagnostic_path = os.path.join(self.output_dir, 'diagnostic_plots.png')
        plt.savefig(diagnostic_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved diagnostic plots: {diagnostic_path}")
        plt.close()
    
    
    def plot_scatter_plots(self, y_pred, y_pred_proba):
        """
        Generate scatter plots showing predictions vs actual values
        
        Args:
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
        """
        print("📊 Generating scatter plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. Prediction Scatter Plot
        # Create sample indices for x-axis
        indices = np.arange(len(self.y_test))
        
        # Scatter plot of actual vs predicted
        colors = ['red' if y_true != y_pred_val else 'green' 
                 for y_true, y_pred_val in zip(self.y_test, y_pred)]
        
        axes[0].scatter(indices, self.y_test, alpha=0.6, s=50, c='blue', label='Actual', marker='o')
        axes[0].scatter(indices, y_pred, alpha=0.6, s=30, c=colors, label='Predicted', marker='x')
        axes[0].set_xlabel('Sample Index', fontsize=12)
        axes[0].set_ylabel('Class (0=Bad, 1=Good)', fontsize=12)
        axes[0].set_title('Actual vs Predicted Labels', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Probability Scatter Plot with True Labels
        axes[1].scatter(indices[self.y_test == 0], y_pred_proba[self.y_test == 0], 
                       alpha=0.6, s=50, c='red', label='Bad Credit (Actual)', marker='o')
        axes[1].scatter(indices[self.y_test == 1], y_pred_proba[self.y_test == 1], 
                       alpha=0.6, s=50, c='green', label='Good Credit (Actual)', marker='s')
        axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
        axes[1].set_xlabel('Sample Index', fontsize=12)
        axes[1].set_ylabel('Predicted Probability', fontsize=12)
        axes[1].set_title('Predicted Probabilities by True Class', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_path = os.path.join(self.output_dir, 'scatter_plots.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved scatter plots: {scatter_path}")
        plt.close()
    
    
    def generate_report(self):
        """
        Generate a comprehensive text report
        """
        print("📝 Generating comprehensive report...")
        
        report_path = os.path.join(self.output_dir, 'training_report.txt')
        
        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("GERMAN CREDIT RISK - ANN MODEL TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("AUTHORS: Devang Thaker & Krish Manvar\n")
            f.write("COURSE: M. Grum: Advanced AI-based Application Systems\n")
            f.write("INSTITUTION: University of Potsdam\n")
            f.write("DATE: January 2026\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("1. MODEL ARCHITECTURE\n")
            f.write("-" * 80 + "\n\n")
            
            # Capture model summary
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = summary_buffer = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            
            f.write(summary_buffer.getvalue())
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("2. TRAINING CONFIGURATION\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Total Epochs Trained: {self.training_metrics['total_epochs_trained']}\n")
            f.write(f"Training Samples: {self.training_metrics['total_training_samples']}\n")
            f.write(f"Test Samples: {self.training_metrics['total_test_samples']}\n")
            f.write(f"Number of Features: {self.training_metrics['number_of_features']}\n")
            f.write(f"Optimizer: Adam (learning_rate=0.001)\n")
            f.write(f"Loss Function: Binary Crossentropy\n")
            f.write(f"Batch Size: 32\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("3. TRAINING RESULTS\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Final Training Loss: {self.training_metrics['final_training_loss']:.6f}\n")
            f.write(f"Final Validation Loss: {self.training_metrics['final_validation_loss']:.6f}\n")
            f.write(f"Final Training Accuracy: {self.training_metrics['final_training_accuracy']:.6f} "
                   f"({self.training_metrics['final_training_accuracy']*100:.2f}%)\n")
            f.write(f"Final Validation Accuracy: {self.training_metrics['final_validation_accuracy']:.6f} "
                   f"({self.training_metrics['final_validation_accuracy']*100:.2f}%)\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("4. TEST SET PERFORMANCE\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Test Accuracy: {self.training_metrics['test_accuracy']:.6f} "
                   f"({self.training_metrics['test_accuracy']*100:.2f}%)\n")
            f.write(f"Test Precision: {self.training_metrics['test_precision']:.6f}\n")
            f.write(f"Test Recall: {self.training_metrics['test_recall']:.6f}\n")
            f.write(f"Test F1-Score: {self.training_metrics['test_f1_score']:.6f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("5. FILES GENERATED\n")
            f.write("-" * 80 + "\n\n")
            f.write("Model Files:\n")
            f.write("  - currentAiSolution.h5 (Keras model)\n")
            f.write("  - currentAiSolution_tf/ (TensorFlow SavedModel)\n")
            f.write("  - currentAiSolution_architecture.json (Model architecture)\n")
            f.write("  - currentAiSolution_weights.h5 (Model weights)\n\n")
            f.write("Metric Files:\n")
            f.write("  - training_metrics.json (Performance metrics)\n")
            f.write("  - training_history.json (Epoch-by-epoch history)\n")
            f.write("  - training_history.csv (History in CSV format)\n\n")
            f.write("Visualization Files:\n")
            f.write("  - training_testing_curves.png (Loss & Accuracy curves)\n")
            f.write("  - diagnostic_plots.png (Confusion Matrix, ROC, PR curves)\n")
            f.write("  - scatter_plots.png (Prediction scatter plots)\n\n")
            f.write("Report Files:\n")
            f.write("  - training_report.txt (This file)\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"   ✓ Saved comprehensive report: {report_path}")
        print()


def print_system_info():
    """
    Print all library versions and system information
    """
    print("\n" + "=" * 70)
    print("📦 SYSTEM & LIBRARY INFORMATION")
    print("=" * 70)
    print()
    
    import sys
    import platform
    
    print("🖥️  System Information:")
    print(f"   Python Version: {sys.version}")
    print(f"   Platform: {platform.platform()}")
    print(f"   Processor: {platform.processor()}")
    print()
    
    print("📚 Library Versions:")
    print(f"   TensorFlow: {tf.__version__}")
    print(f"   Keras: {keras.__version__}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Pandas: {pd.__version__}")
    print(f"   Matplotlib: {plt.matplotlib.__version__}")
    print(f"   Seaborn: {sns.__version__}")
    print(f"   Scikit-learn: {__import__('sklearn').__version__}")
    print()
    
    # Check GPU availability
    print("🎮 GPU Information:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   Available GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu}")
    else:
        print("   No GPU available (using CPU)")
    print()
    
    print("=" * 70)
    print()


def main():
    """
    Main execution function
    """
    # Print system information
    print_system_info()
    
    # Initialize model
    ann = GermanCreditANN(output_dir='images/learningBase')
    
    # Load data
    ann.load_data(
        train_path='data/training_data.csv',
        test_path='data/test_data.csv'
    )
    
    # Build model
    ann.build_model()
    
    # Train model
    ann.train_model(epochs=100, batch_size=32)
    
    # Evaluate model
    y_pred, y_pred_proba = ann.evaluate_model()
    
    # Save model
    ann.save_model()
    
    # Save metrics
    ann.save_metrics()
    
    # Generate visualizations
    ann.plot_training_curves()
    ann.plot_diagnostic_plots(y_pred, y_pred_proba)
    ann.plot_scatter_plots(y_pred, y_pred_proba)
    
    # Generate report
    ann.generate_report()
    
    print("=" * 70)
    print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("📁 All outputs saved in 'learningBase/' directory:")
    print("   ✓ Trained model (currentAiSolution.*)")
    print("   ✓ Training metrics and history")
    print("   ✓ Training/testing curves")
    print("   ✓ Diagnostic plots")
    print("   ✓ Scatter plots")
    print("   ✓ Comprehensive report")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()