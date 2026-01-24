"""
OLS Model Training Script for Flower Classification
Uses PCA for dimensionality reduction and MNLogit (OLS)
Authors: Devang Thaker & Krish Manvar
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Statsmodels for OLS
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===============================================
# CONFIGURATION
# ===============================================
TRAIN_DATA_PATH = "images/learningBase/data/train/training_data.csv"
TEST_DATA_PATH = "images/learningBase/data/validation/test_data.csv"
OUTPUT_DIR = "olsBase"
MODEL_FILE = os.path.join(OUTPUT_DIR, "currentOlsSolution.pkl")
REPORT_FILE = os.path.join(OUTPUT_DIR, "ols_training_report.txt")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")
N_COMPONENTS = 100  # PCA components

FLOWER_CLASSES = {0: 'sunflower', 1: 'daisy', 2: 'dandelion', 3: 'rose', 4: 'tulip'}

# Create directories if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

print("="*70)
print("🌸 FLOWER CLASSIFICATION OLS TRAINING SCRIPT")
print("="*70)

# ===============================================
# STEP 1: LOAD DATA
# ===============================================
print("STEP 1: LOADING DATA")
try:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError("Training or Test CSV files not found. Check paths.")

X_train = train_df.drop(['filename', 'flower_type', 'label'], axis=1).values
y_train = train_df['label'].values.astype(int)
X_test = test_df.drop(['filename', 'flower_type', 'label'], axis=1).values
y_test = test_df['label'].values.astype(int)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Feature dimension: {X_train.shape[1]}")

# Class distribution
for class_id, class_name in FLOWER_CLASSES.items():
    print(f"Class {class_id} ({class_name}): {(y_train==class_id).sum()} samples")

# ===============================================
# STEP 2: FEATURE SCALING & PCA
# ===============================================
print("STEP 2: SCALING & PCA")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=min(N_COMPONENTS, X_train_scaled.shape[1]), random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_variance = pca.explained_variance_ratio_.sum()
print(f"PCA complete. Explained variance: {explained_variance*100:.2f}%")

# ===============================================
# STEP 3: TRAIN MNLOGIT MODEL
# ===============================================
print("STEP 3: TRAINING MNLogit MODEL")
X_train_const = sm.add_constant(X_train_pca)
X_test_const = sm.add_constant(X_test_pca)

model = MNLogit(y_train, X_train_const)
start_time = datetime.now()
result = model.fit(method='newton', maxiter=100, disp=False)
end_time = datetime.now()
training_duration = (end_time - start_time).total_seconds()
print(f"Training completed in {training_duration:.2f}s")

# ===============================================
# STEP 4: EVALUATE MODEL
# ===============================================
y_train_pred = result.predict(X_train_const).argmax(axis=1)
y_test_pred = result.predict(X_test_const).argmax(axis=1)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

class_names = [FLOWER_CLASSES[i] for i in range(len(FLOWER_CLASSES))]
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_test_pred, average=None)

# ===============================================
# STEP 5: SAVE MODEL
# ===============================================
model_data = {
    'model_result': result,
    'scaler': scaler,
    'pca': pca,
    'flower_classes': FLOWER_CLASSES,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'training_duration': training_duration,
    'n_components': N_COMPONENTS,
    'explained_variance': explained_variance,
    'timestamp': datetime.now().isoformat()
}
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model_data, f)
print(f"Model saved: {MODEL_FILE}")

# ===============================================
# STEP 6: VISUALIZATIONS
# ===============================================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('OLS Model - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'confusion_matrix.png'))
plt.close()

# Accuracy comparison
fig, ax = plt.subplots()
ax.bar(['Train', 'Test'], [train_accuracy*100, test_accuracy*100], color=['#4ECDC4', '#FF6B6B'])
ax.set_ylabel('Accuracy (%)')
ax.set_ylim([0, 100])
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'accuracy_comparison.png'))
plt.close()

print(f"Visualizations saved to {VIZ_DIR}")

# ===============================================
# STEP 7: SAVE REPORT
# ===============================================
report_content = f"""
FLOWER CLASSIFICATION OLS - TRAINING REPORT
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training Duration: {training_duration:.2f}s

Training Accuracy: {train_accuracy*100:.2f}%
Test Accuracy: {test_accuracy*100:.2f}%
Explained Variance (PCA): {explained_variance*100:.2f}%
Number of Classes: {len(FLOWER_CLASSES)}
Classes: {', '.join([f'{k}={v}' for k,v in FLOWER_CLASSES.items()])}

PER-CLASS METRICS:
{'Class':<15}{'Precision':<10}{'Recall':<10}{'F1':<10}{'Support':<10}
"""
for i, cname in enumerate(class_names):
    report_content += f"{cname:<15}{precision[i]:<10.2f}{recall[i]:<10.2f}{f1[i]:<10.2f}{support[i]:<10}\n"

with open(REPORT_FILE, 'w') as f:
    f.write(report_content)
print(f"Report saved: {REPORT_FILE}")

print("✅ OLS MODEL TRAINING COMPLETED SUCCESSFULLY!")
