"""
Data Preparation Script for German Credit Risk Dataset
Author: [Krish Manvar]
Date: January 2026
Course: M. Grum: Advanced AI-based Application Systems

This script performs:
1. Data cleaning (handling missing values, outliers)
2. Data normalization
3. Data splitting (80% train, 20% test)
4. Creating activation data sample
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreparator:
    """
    Class to handle all data preparation tasks
    """
    
    def __init__(self, input_file='/data/german_credit_raw.csv'):
        """
        Initialize the DataPreparator
        
        Args:
            input_file (str): Path to the raw data file
        """
        self.input_file = input_file
        self.df = None
        self.df_clean = None
        self.train_df = None
        self.test_df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        print("🔧 DataPreparator initialized")
    
    
    def load_data(self):
        """
        Load the raw data from CSV file
        """
        print("\n📂 Loading raw data...")
        
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Display basic info
            print("\n📋 Columns:", list(self.df.columns))
            print(f"\n🔍 First few rows:")
            print(self.df.head(3))
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: File not found - {self.input_file}")
            print("💡 Please run scraper.py first to download the data")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    
    def clean_data(self):
        """
        Clean the data:
        - Handle missing values
        - Remove duplicates
        - Fix data types
        """
        print("\n🧹 Cleaning data...")
        
        self.df_clean = self.df.copy()
        initial_shape = self.df_clean.shape
        
        # 1. Check for duplicates
        duplicates = self.df_clean.duplicated().sum()
        if duplicates > 0:
            print(f"  • Removing {duplicates} duplicate rows...")
            self.df_clean = self.df_clean.drop_duplicates()
        else:
            print("  ✅ No duplicates found")
        
        # 2. Handle missing values
        print("\n  🔍 Checking for missing values...")
        missing_before = self.df_clean.isnull().sum()
        
        if missing_before.sum() > 0:
            print("  ⚠️  Missing values found:")
            for col in missing_before[missing_before > 0].index:
                print(f"    - {col}: {missing_before[col]} missing")
            
            # Strategy: Fill or drop based on column type
            for col in self.df_clean.columns:
                if self.df_clean[col].isnull().sum() > 0:
                    if self.df_clean[col].dtype in ['float64', 'int64']:
                        # Fill numeric columns with median
                        median_val = self.df_clean[col].median()
                        self.df_clean[col].fillna(median_val, inplace=True)
                        print(f"    ✓ Filled {col} with median: {median_val}")
                    else:
                        # Fill categorical columns with mode
                        mode_val = self.df_clean[col].mode()[0] if len(self.df_clean[col].mode()) > 0 else 'Unknown'
                        self.df_clean[col].fillna(mode_val, inplace=True)
                        print(f"    ✓ Filled {col} with mode: {mode_val}")
        else:
            print("  ✅ No missing values found")
        
        # 3. Verify data types
        print("\n  📊 Data types:")
        for col in self.df_clean.columns:
            print(f"    • {col}: {self.df_clean[col].dtype}")
        
        print(f"\n✅ Data cleaning completed!")
        print(f"   Shape: {initial_shape} → {self.df_clean.shape}")
        
        return self.df_clean
    
    
    def remove_outliers(self, method='iqr', threshold=1.5):
        """
        Remove outliers using IQR method (Interquartile Range)
        
        Args:
            method (str): Method to use ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
        """
        print(f"\n🎯 Removing outliers using {method.upper()} method...")
        
        initial_rows = len(self.df_clean)
        
        # Get numeric columns only
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.df_clean[col].quantile(0.25)
                Q3 = self.df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers_count = ((self.df_clean[col] < lower_bound) | 
                                 (self.df_clean[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    print(f"  • {col}: Found {outliers_count} outliers")
                    self.df_clean = self.df_clean[
                        (self.df_clean[col] >= lower_bound) & 
                        (self.df_clean[col] <= upper_bound)
                    ]
        
        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(self.df_clean[col]))
                outliers_count = (z_scores > 3).sum()
                
                if outliers_count > 0:
                    print(f"  • {col}: Found {outliers_count} outliers")
                    self.df_clean = self.df_clean[z_scores <= 3]
        
        final_rows = len(self.df_clean)
        removed = initial_rows - final_rows
        
        print(f"\n✅ Outlier removal completed!")
        print(f"   Rows: {initial_rows} → {final_rows} (Removed: {removed})")
        
        return self.df_clean
    
    
    def normalize_data(self):
        """
        Normalize/standardize numerical features
        Also encode categorical features
        """
        print("\n📏 Normalizing and encoding data...")
        
        # Separate numeric and categorical columns
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df_clean.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  📊 Numeric columns ({len(numeric_cols)}): {numeric_cols}")
        print(f"  📝 Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        
        # Create a copy for normalized data
        df_normalized = self.df_clean.copy()
        
        # 1. Encode categorical variables
        if len(categorical_cols) > 0:
            print("\n  🔤 Encoding categorical variables...")
            for col in categorical_cols:
                # FIX: Replace string 'nan' with actual NaN, then fill with mode
                df_normalized[col] = df_normalized[col].replace('nan', np.nan)
                
                # Fill any remaining NaN values with mode
                if df_normalized[col].isnull().sum() > 0:
                    mode_val = df_normalized[col].mode()[0] if len(df_normalized[col].mode()) > 0 else 'Unknown'
                    df_normalized[col].fillna(mode_val, inplace=True)
                    print(f"    ℹ️  Filled remaining NaN in {col} with: {mode_val}")
                
                # Now encode
                le = LabelEncoder()
                df_normalized[col + '_encoded'] = le.fit_transform(df_normalized[col].astype(str))
                self.label_encoders[col] = le
                
                # Show mapping (use the classes from the encoder, not the dataframe)
                print(f"    • {col}:")
                for i, val in enumerate(le.classes_[:5]):
                    print(f"      '{val}' → {i}")
                if len(le.classes_) > 5:
                    print(f"      ... and {len(le.classes_) - 5} more")
        
        # 2. Normalize numeric columns
        if len(numeric_cols) > 0:
            print("\n  📏 Standardizing numeric features...")
            
            # Fit the scaler on numeric columns
            df_normalized[numeric_cols] = self.scaler.fit_transform(df_normalized[numeric_cols])
            
            print(f"    ✓ Normalized {len(numeric_cols)} numeric columns")
            print(f"    ℹ️  Mean ≈ 0, Std ≈ 1 for all numeric features")
        
        self.df_clean = df_normalized
        
        print("\n✅ Normalization completed!")
        
        return self.df_clean
    
    
    def save_joint_data(self, output_dir='data'):
        """
        Save the cleaned and normalized joint dataset
        
        Args:
            output_dir (str): Directory to save the data
        """
        print("\n💾 Saving joint data collection...")
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'joint_data_collection.csv')
        
        try:
            self.df_clean.to_csv(output_path, index=False)
            print(f"✅ Joint data saved: {output_path}")
            print(f"   Size: {self.df_clean.shape[0]} rows × {self.df_clean.shape[1]} columns")
            return output_path
        except Exception as e:
            print(f"❌ Error saving joint data: {e}")
            return None
    
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training (80%) and testing (20%) sets
        
        Args:
            test_size (float): Proportion of test data (default: 0.2)
            random_state (int): Random seed for reproducibility
        """
        print(f"\n✂️  Splitting data (Train: {int((1-test_size)*100)}%, Test: {int(test_size*100)}%)...")
        
        # Split the data
        self.train_df, self.test_df = train_test_split(
            self.df_clean,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"✅ Data split completed!")
        print(f"   Training set: {self.train_df.shape[0]} rows")
        print(f"   Test set: {self.test_df.shape[0]} rows")
        print(f"   Ratio: {self.train_df.shape[0]/self.df_clean.shape[0]:.1%} / {self.test_df.shape[0]/self.df_clean.shape[0]:.1%}")
        
        return self.train_df, self.test_df
    
    
    def save_split_data(self, output_dir='data'):
        """
        Save training and test datasets separately
        
        Args:
            output_dir (str): Directory to save the data
        """
        print("\n💾 Saving training and test data...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training data
        train_path = os.path.join(output_dir, 'training_data.csv')
        self.train_df.to_csv(train_path, index=False)
        print(f"✅ Training data saved: {train_path}")
        print(f"   Size: {self.train_df.shape[0]} rows × {self.train_df.shape[1]} columns")
        
        # Save test data
        test_path = os.path.join(output_dir, 'test_data.csv')
        self.test_df.to_csv(test_path, index=False)
        print(f"✅ Test data saved: {test_path}")
        print(f"   Size: {self.test_df.shape[0]} rows × {self.test_df.shape[1]} columns")
        
        return train_path, test_path
    
    
    def create_activation_data(self, output_dir='data'):
        """
        Create activation_data.csv with one sample from test data
        
        Args:
            output_dir (str): Directory to save the data
        """
        print("\n🎯 Creating activation data (single test sample)...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get one random sample from test data
        activation_sample = self.test_df.sample(n=1, random_state=42)
        
        # Save activation data
        activation_path = os.path.join(output_dir, 'activation_data.csv')
        activation_sample.to_csv(activation_path, index=False)
        
        print(f"✅ Activation data saved: {activation_path}")
        print(f"   Size: 1 row × {activation_sample.shape[1]} columns")
        print(f"\n📋 Sample data:")
        print(activation_sample.to_string())
        
        return activation_path
    
    
    def generate_report(self):
        """
        Generate a summary report of the data preparation process
        """
        print("\n" + "="*70)
        print("📊 DATA PREPARATION SUMMARY REPORT")
        print("="*70)
        
        print(f"\n1️⃣  ORIGINAL DATA:")
        print(f"   • Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        print(f"\n2️⃣  CLEANED & NORMALIZED DATA:")
        print(f"   • Shape: {self.df_clean.shape[0]} rows × {self.df_clean.shape[1]} columns")
        print(f"   • Data quality: ✅ No missing values, ✅ No duplicates")
        
        print(f"\n3️⃣  DATA SPLIT:")
        print(f"   • Training data: {self.train_df.shape[0]} rows ({self.train_df.shape[0]/self.df_clean.shape[0]:.1%})")
        print(f"   • Test data: {self.test_df.shape[0]} rows ({self.test_df.shape[0]/self.df_clean.shape[0]:.1%})")
        print(f"   • Activation data: 1 row (single test sample)")
        
        print(f"\n4️⃣  FILES CREATED:")
        print(f"   ✓ joint_data_collection.csv")
        print(f"   ✓ training_data.csv")
        print(f"   ✓ test_data.csv")
        print(f"   ✓ activation_data.csv")
        
        print("\n" + "="*70)


def main():
    """
    Main function to execute the complete data preparation pipeline
    """
    print("\n" + "="*70)
    print("🚀 GERMAN CREDIT RISK - DATA PREPARATION PIPELINE")
    print("="*70)
    
    # Initialize preparator
    preparator = DataPreparator(input_file='data/german_credit_raw.csv')
    
    # Step 1: Load data
    if not preparator.load_data():
        return
    
    # Step 2: Clean data
    preparator.clean_data()
    
    # Step 3: Remove outliers
    preparator.remove_outliers(method='iqr', threshold=1.5)
    
    # Step 4: Normalize data
    preparator.normalize_data()
    
    # Step 5: Save joint data
    preparator.save_joint_data()
    
    # Step 6: Split data (80-20)
    preparator.split_data(test_size=0.2, random_state=42)
    
    # Step 7: Save split data
    preparator.save_split_data()
    
    # Step 8: Create activation data
    preparator.create_activation_data()
    
    # Step 9: Generate report
    preparator.generate_report()
    
    print("\n✅ DATA PREPARATION COMPLETED SUCCESSFULLY! ✅")
    print("\n💡 Next steps:")
    print("   1. Check the data/ folder for all generated CSV files")
    print("   2. Use training_data.csv for model training")
    print("   3. Use test_data.csv for model evaluation")
    print("   4. Use activation_data.csv for single prediction testing")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()