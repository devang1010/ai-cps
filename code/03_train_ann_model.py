"""
Flower Classification - ANN Model Training Script (FIXED VERSION)
Uses TensorFlow/Keras to train flower classification model
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Scikit-learn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("🌸 FLOWER CLASSIFICATION ANN TRAINING SCRIPT")
print("=" * 70)
print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print("=" * 70)


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data(train_path, test_path):
    """Load and prepare training and test data"""
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    try:
        # Load CSV files
        print(f"\n📂 Loading training data from: {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"   ✓ Loaded {len(train_df)} training samples")
        
        print(f"\n📂 Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path)
        print(f"   ✓ Loaded {len(test_df)} test samples")
        
        # Check columns
        print(f"\n📊 Dataset columns: {len(train_df.columns)}")
        print(f"   First few columns: {list(train_df.columns[:5])}")
        
        # Separate features and labels
        pixel_cols = [col for col in train_df.columns if col.startswith('pixel_')]
        print(f"\n🎨 Found {len(pixel_cols)} pixel features")
        
        X_train = train_df[pixel_cols].values.astype('float32')
        y_train = train_df['label'].values.astype('int32')
        
        X_test = test_df[pixel_cols].values.astype('float32')
        y_test = test_df['label'].values.astype('int32')
        
        # Data info
        print(f"\n✅ Data shapes:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   X_test: {X_test.shape}")
        print(f"   y_test: {y_test.shape}")
        
        # Check value ranges
        print(f"\n📈 Data statistics:")
        print(f"   X_train min/max: {X_train.min():.4f} / {X_train.max():.4f}")
        print(f"   y_train unique values: {np.unique(y_train)}")
        
        # Class distribution
        print(f"\n🌸 Class distribution (training):")
        flower_names = ['sunflower', 'daisy', 'dandelion', 'rose', 'tulip']
        for i in range(5):
            count = np.sum(y_train == i)
            print(f"   {i} ({flower_names[i]}): {count} samples")
        
        return X_train, y_train, X_test, y_test, flower_names
    
    except Exception as e:
        print(f"\n❌ ERROR loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# STEP 2: BUILD MODEL
# ============================================================================

def build_model(input_dim, num_classes):
    """Build ANN architecture"""
    print("\n" + "=" * 70)
    print("STEP 2: BUILDING MODEL")
    print("=" * 70)
    
    model = Sequential([
        # Input + Hidden Layer 1
        Dense(512, activation='relu', input_shape=(input_dim,), name='dense_1'),
        BatchNormalization(name='bn_1'),
        Dropout(0.4, name='dropout_1'),
        
        # Hidden Layer 2
        Dense(256, activation='relu', name='dense_2'),
        BatchNormalization(name='bn_2'),
        Dropout(0.3, name='dropout_2'),
        
        # Hidden Layer 3
        Dense(128, activation='relu', name='dense_3'),
        BatchNormalization(name='bn_3'),
        Dropout(0.3, name='dropout_3'),
        
        # Hidden Layer 4
        Dense(64, activation='relu', name='dense_4'),
        Dropout(0.2, name='dropout_4'),
        
        # Output Layer
        Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',  # No need for one-hot encoding
        metrics=['accuracy']
    )
    
    print("\n📋 Model Architecture:")
    model.summary()
    
    print(f"\n📊 Total parameters: {model.count_params():,}")
    
    return model


# ============================================================================
# STEP 3: TRAIN MODEL
# ============================================================================

def train_model(model, X_train, y_train, X_test, y_test, output_dir, epochs=100):
    """Train the model"""
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING MODEL")
    print("=" * 70)
    
    print(f"\n⚙️  Training configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: 32")
    print(f"   Optimizer: Adam (lr=0.001)")
    print(f"   Loss: Sparse Categorical Crossentropy")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'currentAiSolution.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    print(f"\n🚀 Starting training...")
    print("=" * 70)
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print("=" * 70)
    print(f"✅ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")
    
    return history, training_time


# ============================================================================
# STEP 4: EVALUATE MODEL
# ============================================================================

def evaluate_model(model, X_test, y_test, flower_names):
    """Evaluate model performance"""
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATING MODEL")
    print("=" * 70)
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📊 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=flower_names, digits=4))
    
    return y_pred, test_loss, test_acc


# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================

def create_visualizations(history, y_test, y_pred, flower_names, output_dir):
    """Create all visualizations"""
    print("\n" + "=" * 70)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Training Curves
    print("\n   📉 Creating training curves...")
    plot_training_curves(history, viz_dir)
    
    # 2. Confusion Matrix
    print("   📊 Creating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, flower_names, viz_dir)
    
    # 3. Classification Metrics
    print("   📈 Creating classification metrics...")
    plot_classification_metrics(y_test, y_pred, flower_names, viz_dir)
    
    # 4. Accuracy Breakdown
    print("   🎯 Creating accuracy breakdown...")
    plot_accuracy_breakdown(y_test, y_pred, flower_names, viz_dir)
    
    print(f"\n✅ All visualizations saved to: {viz_dir}/")


def plot_training_curves(history, viz_dir):
    """Plot loss and accuracy curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0].plot(epochs, history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    axes[1].plot(epochs, history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_test, y_pred, flower_names, viz_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=flower_names, yticklabels=flower_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_classification_metrics(y_test, y_pred, flower_names, viz_dir):
    """Plot precision, recall, F1-score"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    x = np.arange(len(flower_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#e74c3c')
    ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')
    
    ax.set_xlabel('Flower Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics by Flower Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(flower_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'classification_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_breakdown(y_test, y_pred, flower_names, viz_dir):
    """Plot per-class accuracy"""
    accuracies = []
    for i in range(len(flower_names)):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(flower_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.xlabel('Flower Type', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'accuracy_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# STEP 6: SAVE REPORT
# ============================================================================

def save_report(history, training_time, test_loss, test_acc, flower_names, output_dir):
    """Save training report"""
    print("\n" + "=" * 70)
    print("STEP 6: SAVING REPORT")
    print("=" * 70)
    
    report_path = os.path.join(output_dir, 'training_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FLOWER CLASSIFICATION ANN - TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Duration: {training_time:.2f}s ({training_time/60:.2f} min)\n\n")
        
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        f.write("Layer 1: Dense(512) + BatchNorm + Dropout(0.4)\n")
        f.write("Layer 2: Dense(256) + BatchNorm + Dropout(0.3)\n")
        f.write("Layer 3: Dense(128) + BatchNorm + Dropout(0.3)\n")
        f.write("Layer 4: Dense(64) + Dropout(0.2)\n")
        f.write("Output: Dense(5) + Softmax\n\n")
        
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Optimizer: Adam (lr=0.001)\n")
        f.write(f"Loss: Sparse Categorical Crossentropy\n")
        f.write(f"Total Epochs: {len(history.history['loss'])}\n")
        f.write(f"Batch Size: 32\n\n")
        
        f.write("FINAL RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Training Loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"Training Accuracy: {history.history['accuracy'][-1]:.6f} ({history.history['accuracy'][-1]*100:.2f}%)\n")
        f.write(f"Validation Loss: {test_loss:.6f}\n")
        f.write(f"Validation Accuracy: {test_acc:.6f} ({test_acc*100:.2f}%)\n\n")
        
        f.write("EPOCH HISTORY\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}\n")
        f.write("-" * 70 + "\n")
        
        for i in range(len(history.history['loss'])):
            f.write(f"{i+1:<8} ")
            f.write(f"{history.history['loss'][i]:<12.6f} ")
            f.write(f"{history.history['accuracy'][i]:<12.6f} ")
            f.write(f"{history.history['val_loss'][i]:<12.6f} ")
            f.write(f"{history.history['val_accuracy'][i]:<12.6f}\n")
    
    # Save history CSV
    history_df = pd.DataFrame({
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_loss': history.history['loss'],
        'train_accuracy': history.history['accuracy'],
        'val_loss': history.history['val_loss'],
        'val_accuracy': history.history['val_accuracy']
    })
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    print(f"\n   ✓ Report saved: {report_path}")
    print(f"   ✓ History CSV saved: training_history.csv")

    
def augment_data(X_train, y_train):
    """Simple data augmentation using noise injection"""
    print("\n🔄 Augmenting training data...")
    
    augmented_X = []
    augmented_y = []
    
    # Original data
    augmented_X.append(X_train)
    augmented_y.append(y_train)
    
    # Add slight noise (brightness variation)
    noise_factor = 0.05
    X_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
    X_noisy = np.clip(X_noisy, 0., 1.)
    augmented_X.append(X_noisy)
    augmented_y.append(y_train)
    
    # Combine
    X_aug = np.vstack(augmented_X)
    y_aug = np.hstack(augmented_y)
    
    print(f"   Original: {X_train.shape[0]} samples")
    print(f"   Augmented: {X_aug.shape[0]} samples")
    
    return X_aug, y_aug

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Configuration
    TRAIN_PATH = 'images/learningBase/data/train/training_data.csv'
    TEST_PATH = 'images/learningBase/data/validation/test_data.csv'
    OUTPUT_DIR = 'learningBase'
    EPOCHS = 100
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run pipeline
    X_train, y_train, X_test, y_test, flower_names = load_data(TRAIN_PATH, TEST_PATH)

    X_train, y_train = augment_data(X_train, y_train)
    
    model = build_model(input_dim=X_train.shape[1], num_classes=5)
    
    history, training_time = train_model(model, X_train, y_train, X_test, y_test, OUTPUT_DIR, EPOCHS)
    
    y_pred, test_loss, test_acc = evaluate_model(model, X_test, y_test, flower_names)
    
    create_visualizations(history, y_test, y_pred, flower_names, OUTPUT_DIR)
    
    save_report(history, training_time, test_loss, test_acc, flower_names, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"   ✓ Model: currentAiSolution.h5")
    print(f"   ✓ Report: training_report.txt")
    print(f"   ✓ History: training_history.csv")
    print(f"   ✓ Visualizations: visualizations/")
    print(f"\n🎯 Final Validation Accuracy: {test_acc*100:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()