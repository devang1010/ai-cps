"""
OLS Model Training for German Credit Risk Classification
Author: Devang Thaker & Krish Manvar
Date: January 2026
Course: M. Grum: Advanced AI-based Application Systems

This script:
1. Loads training and test data
2. Builds an OLS model using Statsmodels
3. Trains and evaluates the model
4. Generates diagnostic and scatter plots
5. Saves the model as a pickle file and performance metrics
6. Compares predictions with ANN model (if available)
7. Generates comparison bar graphs between OLS and ANN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Print library versions at the start
print("="*70)
print("📦 LIBRARY VERSIONS")
print("="*70)
import sys, platform
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Seaborn: {sns.__version__}")
print(f"Statsmodels: {sm.__version__}")
print("="*70, "\n")


class GermanCreditOLS:
    """
    OLS Model for German Credit Risk Classification
    """
    def __init__(self, output_dir='images/olsBase'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.model = None
        self.metrics = {}
    
    def load_data(self, train_path='images/learningBase/data/training_data.csv',
                  test_path='images/learningBase/data/test_data.csv'):
        """Load and preprocess data"""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        target_col = 'Risk'
        leakage_cols = ['Risk_encoded', 'Unnamed: 0']
        train_df = train_df.drop(columns=[c for c in leakage_cols if c in train_df.columns])
        test_df = test_df.drop(columns=[c for c in leakage_cols if c in test_df.columns])
        
        # Separate features and target
        self.X_train = train_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        self.y_train = train_df[target_col].map({'bad':0,'good':1}).astype(int)
        
        self.X_test = test_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        self.y_test = test_df[target_col].map({'bad':0,'good':1}).astype(int)
        
        print(f"Loaded training samples: {len(self.X_train)}; test samples: {len(self.X_test)}")
    
    def train_model(self):
        """Train OLS using statsmodels"""
        X_train_const = sm.add_constant(self.X_train)
        self.model = sm.OLS(self.y_train, X_train_const).fit()
        print(self.model.summary())
    
    def evaluate_model(self):
        """Evaluate OLS and generate predictions"""
        X_test_const = sm.add_constant(self.X_test)
        y_pred_proba = self.model.predict(X_test_const)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        self.metrics = {
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'test_precision': precision_score(self.y_test, y_pred),
            'test_recall': recall_score(self.y_test, y_pred),
            'test_f1': f1_score(self.y_test, y_pred)
        }
        
        print(f"OLS Test Accuracy: {self.metrics['test_accuracy']:.4f}")
        return y_pred, y_pred_proba
    
    def save_model(self):
        """Save model using pickle"""
        model_path = os.path.join(self.output_dir, 'currentOlsSolution.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Saved OLS model: {model_path}")
    
    def save_metrics(self):
        """Save metrics as JSON"""
        metrics_path = os.path.join(self.output_dir, 'ols_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Saved metrics: {metrics_path}")
    
    def plot_diagnostics(self, y_pred, y_pred_proba):
        """Generate diagnostic plots similar to ANN"""
        fig, axes = plt.subplots(1, 2, figsize=(15,5))
        
        # 1. Residual Plot
        residuals = self.y_test - y_pred_proba
        axes[0].scatter(y_pred_proba, residuals)
        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].set_xlabel('Predicted Probabilities')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('OLS Residual Plot')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Predicted vs Actual
        axes[1].scatter(range(len(self.y_test)), self.y_test, label='Actual', color='blue', alpha=0.6)
        axes[1].scatter(range(len(self.y_test)), y_pred, label='Predicted', color='green', alpha=0.6)
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Class (0=Bad,1=Good)')
        axes[1].set_title('OLS Actual vs Predicted')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'ols_diagnostic_scatter.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved diagnostic & scatter plots: {plot_path}")

    def plot_comparison_bars(self, ann_metrics):
        """Generate bar chart comparison between OLS and ANN"""
        if not ann_metrics:
            print("No ANN metrics available for comparison.")
            return
        
        # Extract metrics for comparison
        metrics_to_compare = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        ols_values = [self.metrics.get(m, 0) for m in metrics_to_compare]
        ann_values = [ann_metrics.get(m, 0) for m in metrics_to_compare]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(metric_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ols_values, width, label='OLS Model', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, ann_values, width, label='ANN Model', color='#e74c3c', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('OLS vs ANN Model Performance Comparison\nGerman Credit Risk Classification', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'ols_vs_ann_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison bar chart: {plot_path}")
    
    def plot_detailed_comparison(self, ann_metrics):
        """Generate detailed comparison with multiple visualizations"""
        if not ann_metrics:
            print("No ANN metrics available for detailed comparison.")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Bar Chart Comparison
        ax1 = fig.add_subplot(gs[0, :])
        metrics_to_compare = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        ols_values = [self.metrics.get(m, 0) for m in metrics_to_compare]
        ann_values = [ann_metrics.get(m, 0) for m in metrics_to_compare]
        
        x = np.arange(len(metric_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, ols_values, width, label='OLS Model', color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, ann_values, width, label='ANN Model', color='#e74c3c', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Performance Metrics', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax1.set_title('OLS vs ANN Performance Comparison', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels, fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.0])
        
        # 2. Difference Plot
        ax2 = fig.add_subplot(gs[1, 0])
        differences = [ols - ann for ols, ann in zip(ols_values, ann_values)]
        colors = ['green' if d > 0 else 'red' for d in differences]
        
        bars = ax2.barh(metric_labels, differences, color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        for i, (bar, diff) in enumerate(zip(bars, differences)):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{diff:+.3f}',
                    ha='left' if width > 0 else 'right',
                    va='center', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Difference (OLS - ANN)', fontsize=10, fontweight='bold')
        ax2.set_title('Performance Difference\n(Positive = OLS Better)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Percentage Comparison Table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('tight')
        ax3.axis('off')
        
        table_data = []
        table_data.append(['Metric', 'OLS', 'ANN', 'Difference'])
        for label, ols_val, ann_val in zip(metric_labels, ols_values, ann_values):
            diff = ols_val - ann_val
            table_data.append([
                label,
                f'{ols_val:.4f}',
                f'{ann_val:.4f}',
                f'{diff:+.4f}'
            ])
        
        table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data)):
            for j in range(4):
                if j == 3:  # Difference column
                    diff_val = float(table_data[i][j])
                    if diff_val > 0:
                        table[(i, j)].set_facecolor('#d5f4e6')
                    elif diff_val < 0:
                        table[(i, j)].set_facecolor('#fadbd8')
                    else:
                        table[(i, j)].set_facecolor('#f8f9fa')
                else:
                    table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
        
        ax3.set_title('Detailed Metrics Comparison', fontsize=11, fontweight='bold', pad=20)
        
        plt.suptitle('OLS vs ANN Model - Comprehensive Comparison\nGerman Credit Risk Classification',
                    fontsize=15, fontweight='bold', y=0.98)
        
        plot_path = os.path.join(self.output_dir, 'ols_vs_ann_detailed_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved detailed comparison chart: {plot_path}")

    def generate_report(self, y_pred, y_pred_proba, ann_metrics=None):
        """Generate OLS report comparing with ANN if available"""
        report_path = os.path.join(self.output_dir, 'ols_training_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GERMAN CREDIT RISK - OLS MODEL REPORT\n")
            f.write("="*80 + "\n\n")
            f.write("AUTHORS: Devang Thaker & Krish Manvar\n")
            f.write("COURSE: M. Grum: Advanced AI-based Application Systems\n")
            f.write("DATE: January 2026\n\n")
        
            f.write("-"*80 + "\n")
            f.write("1. MODEL COEFFICIENTS\n")
            f.write("-"*80 + "\n")
            f.write(str(self.model.params))
            f.write("\n\n")
        
            f.write("-"*80 + "\n")
            f.write("2. TEST PERFORMANCE\n")
            f.write("-"*80 + "\n")
            for key, value in self.metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
        
            if ann_metrics:
                f.write("-"*80 + "\n")
                f.write("3. COMPARISON WITH ANN\n")
                f.write("-"*80 + "\n")
                for key in ['test_accuracy','test_precision','test_recall','test_f1']:
                    ols_val = self.metrics[key]
                    ann_val = ann_metrics.get(key, 0)
                    diff = ols_val - ann_val
                    winner = "OLS" if diff > 0 else "ANN" if diff < 0 else "TIE"
                    f.write(f"{key}:\n")
                    f.write(f"  OLS: {ols_val:.4f}\n")
                    f.write(f"  ANN: {ann_val:.4f}\n")
                    f.write(f"  Difference: {diff:+.4f} (Winner: {winner})\n\n")
        
            f.write("-"*80 + "\n")
            f.write("4. GENERATED FILES\n")
            f.write("-"*80 + "\n")
            f.write("Model files:\n")
            f.write("  - currentOlsSolution.pkl\n")
            f.write("  - currentOlsSolution.xml\n")
            f.write("Metric files:\n")
            f.write("  - ols_metrics.json\n")
            f.write("Visualization files:\n")
            f.write("  - ols_diagnostic_scatter.png\n")
            f.write("  - ols_vs_ann_comparison.png\n")
            f.write("  - ols_vs_ann_detailed_comparison.png\n")
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Saved OLS report: {report_path}")


    def save_xml_model(self):
        """Save simple XML with coefficients"""
        import xml.etree.ElementTree as ET
        root = ET.Element("OLSModel")
        for name, coef in self.model.params.items():
            param = ET.SubElement(root, "Coefficient", name=str(name))
            param.text = str(coef)
        tree = ET.ElementTree(root)
        xml_path = os.path.join(self.output_dir, "currentOlsSolution.xml")
        tree.write(xml_path)
        print(f"Saved OLS XML model: {xml_path}")



def main():
    # Initialize OLS model
    ols = GermanCreditOLS(output_dir='images/olsBase')
    
    # Load data
    ols.load_data()
    
    # Train model
    ols.train_model()
    
    # Evaluate model
    y_pred, y_pred_proba = ols.evaluate_model()
    
    # Save model and metrics
    ols.save_model()
    ols.save_metrics()
    ols.save_xml_model()
    
    # Generate visualizations
    ols.plot_diagnostics(y_pred, y_pred_proba)
    
    # Try to load ANN metrics for comparison
    ann_metrics = None
    ann_metrics_path = 'images/learningBase/training_metrics.json'
    if os.path.exists(ann_metrics_path):
        with open(ann_metrics_path, 'r') as f:
            ann_metrics = json.load(f)
        print(f"✅ Loaded ANN metrics for comparison from: {ann_metrics_path}")
        
        # Generate comparison visualizations
        ols.plot_comparison_bars(ann_metrics)
        ols.plot_detailed_comparison(ann_metrics)
    else:
        print(f"⚠️  ANN metrics not found at {ann_metrics_path}. Comparison charts will not be generated.")
    
    # Generate comprehensive report
    ols.generate_report(y_pred, y_pred_proba, ann_metrics)
    
    print("\n" + "="*70)
    print("✅ OLS WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*70)
    print("All files saved in 'images/olsBase/':")
    print("  📁 Model files: currentOlsSolution.pkl, currentOlsSolution.xml")
    print("  📊 Metrics: ols_metrics.json")
    print("  📈 Visualizations:")
    print("     - ols_diagnostic_scatter.png")
    if ann_metrics:
        print("     - ols_vs_ann_comparison.png")
        print("     - ols_vs_ann_detailed_comparison.png")
    print("  📄 Report: ols_training_report.txt")
    print("="*70)


if __name__ == "__main__":
    main()