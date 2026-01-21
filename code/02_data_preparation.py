"""
Data Preparation Script for Flower Classification
Converts scraped images into CSV format with train/test split
"""
 
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import time

class FlowerDataPreparation:
    def __init__(self, data_dir='data/flowers', output_dir='data'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.flower_types = ['sunflower', 'daisy', 'dandelion', 'rose', 'tulip']
        self.img_size = (64, 64)  # Resize to 64x64 for manageable CSV size
        
    def load_and_process_images(self):
        """Load all images and convert to feature vectors"""
        print("=" * 60)
        print("📊 LOADING AND PROCESSING IMAGES")
        print("=" * 60)
        
        all_data = []
        total_processed = 0
        total_failed = 0
        
        for idx, flower in enumerate(self.flower_types):
            folder_path = os.path.join(self.data_dir, flower)
            
            if not os.path.exists(folder_path):
                print(f"\n⚠ Warning: Folder not found: {folder_path}")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            
            print(f"\n🌸 Processing {flower}: {len(image_files)} images found")
            
            processed = 0
            failed = 0
            
            for img_file in image_files:
                try:
                    img_path = os.path.join(folder_path, img_file)
                    
                    # Load image
                    img = Image.open(img_path)
                    
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize image to standard size
                    img = img.resize(self.img_size)
                    
                    # Convert to numpy array and flatten
                    img_array = np.array(img).flatten()
                    
                    # Normalize pixel values to [0, 1]
                    img_array = img_array / 255.0
                    
                    # Create data entry
                    data_entry = {
                        'filename': img_file,
                        'flower_type': flower,
                        'label': idx,
                        'pixel_data': img_array.tolist()
                    }
                    
                    all_data.append(data_entry)
                    processed += 1
                    
                    # Progress indicator
                    if processed % 10 == 0:
                        print(f"  ✓ Processed {processed}/{len(image_files)} images", end='\r')
                    
                except Exception as e:
                    print(f"\n  ✗ Error processing {img_file}: {str(e)}")
                    failed += 1
                    continue
            
            print(f"\n  ✓ Successfully processed: {processed} {flower} images")
            if failed > 0:
                print(f"  ✗ Failed: {failed} images")
            
            total_processed += processed
            total_failed += failed
        
        print(f"\n{'='*60}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"Total images processed: {total_processed}")
        print(f"Total images failed: {total_failed}")
        print(f"{'='*60}")
        
        return all_data
    
    def remove_outliers(self, data):
        """Remove outliers based on image statistics"""
        print("\n🔍 REMOVING OUTLIERS...")
        
        cleaned_data = []
        removed_count = 0
        
        for entry in data:
            pixels = np.array(entry['pixel_data'])
            
            # Calculate statistics
            mean_val = np.mean(pixels)
            std_val = np.std(pixels)
            
            # Remove images that are too dark, too bright, or too uniform
            if 0.05 < mean_val < 0.95 and std_val > 0.02:
                cleaned_data.append(entry)
            else:
                removed_count += 1
        
        print(f"  ✓ Removed {removed_count} outlier images")
        print(f"  ✓ Remaining images: {len(cleaned_data)}")
        
        return cleaned_data
    
    def create_csv_files(self, data):
        """Create joint, training, test, and activation CSV files"""
        print("\n📁 CREATING CSV FILES...")
        
        if len(data) == 0:
            print("✗ No data to process!")
            return None, None, None
        
        # Create DataFrame
        print("  → Building DataFrame...")
        rows = []
        
        for entry in data:
            row = {
                'filename': entry['filename'],
                'flower_type': entry['flower_type'],
                'label': entry['label']
            }
            
            # Add pixel values as separate columns
            for i, pixel in enumerate(entry['pixel_data']):
                row[f'pixel_{i}'] = pixel
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 1. Save joint data collection
        print("\n  → Saving joint_data_collection.csv...")
        joint_path = os.path.join(self.output_dir, 'joint_data_collection.csv')
        df.to_csv(joint_path, index=False)
        file_size_mb = os.path.getsize(joint_path) / (1024 * 1024)
        print(f"  ✓ Created: joint_data_collection.csv")
        print(f"    - Entries: {len(df)}")
        print(f"    - Size: {file_size_mb:.2f} MB")
        
        # 2. Split into training (80%) and test (20%)
        print("\n  → Splitting data (80/20)...")
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['label'], 
            random_state=42
        )
        
        # Save training data
        print("  → Saving training_data.csv...")
        train_path = os.path.join(self.output_dir, 'training_data.csv')
        train_df.to_csv(train_path, index=False)
        file_size_mb = os.path.getsize(train_path) / (1024 * 1024)
        print(f"  ✓ Created: training_data.csv")
        print(f"    - Entries: {len(train_df)} (80%)")
        print(f"    - Size: {file_size_mb:.2f} MB")
        
        # Save test data
        print("  → Saving test_data.csv...")
        test_path = os.path.join(self.output_dir, 'test_data.csv')
        test_df.to_csv(test_path, index=False)
        file_size_mb = os.path.getsize(test_path) / (1024 * 1024)
        print(f"  ✓ Created: test_data.csv")
        print(f"    - Entries: {len(test_df)} (20%)")
        print(f"    - Size: {file_size_mb:.2f} MB")
        
        # 3. Create activation data (one sample from each class)
        print("\n  → Creating activation_data.csv...")
        activation_samples = []
        for label in sorted(df['label'].unique()):
            sample = test_df[test_df['label'] == label].head(1)
            activation_samples.append(sample)
        
        activation_df = pd.concat(activation_samples, ignore_index=True)
        activation_path = os.path.join(self.output_dir, 'activation_data.csv')
        activation_df.to_csv(activation_path, index=False)
        print(f"  ✓ Created: activation_data.csv")
        print(f"    - Entries: {len(activation_df)} (1 per class)")
        
        # Print data distribution
        print("\n" + "=" * 60)
        print("📊 DATA DISTRIBUTION")
        print("=" * 60)
        
        print("\n📈 Training Set Distribution:")
        print(train_df['flower_type'].value_counts().to_string())
        
        print("\n📈 Test Set Distribution:")
        print(test_df['flower_type'].value_counts().to_string())
        
        print("\n📈 Activation Set:")
        print(activation_df['flower_type'].value_counts().to_string())
        
        return train_df, test_df, activation_df
    
    def generate_metadata(self, train_df, test_df):
        """Generate metadata file with dataset information"""
        print("\n📄 GENERATING METADATA...")
        
        metadata_path = os.path.join(self.output_dir, 'dataset_metadata.txt')
        
        total_images = len(train_df) + len(test_df)
        feature_count = self.img_size[0] * self.img_size[1] * 3  # RGB channels
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FLOWER CLASSIFICATION DATASET METADATA\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Images: {total_images}\n")
            f.write(f"Training Images: {len(train_df)} (80%)\n")
            f.write(f"Test Images: {len(test_df)} (20%)\n")
            f.write(f"Activation Samples: 5 (1 per class)\n\n")
            
            f.write("FLOWER CLASSES\n")
            f.write("-" * 60 + "\n")
            for i, flower in enumerate(self.flower_types):
                count = len(train_df[train_df['label'] == i]) + len(test_df[test_df['label'] == i])
                f.write(f"Label {i}: {flower.capitalize()} ({count} images)\n")
            
            f.write("\n")
            f.write("IMAGE PROCESSING\n")
            f.write("-" * 60 + "\n")
            f.write(f"Original Image Size: Variable\n")
            f.write(f"Processed Image Size: {self.img_size[0]}x{self.img_size[1]} pixels\n")
            f.write(f"Color Space: RGB (3 channels)\n")
            f.write(f"Pixel Normalization: [0, 1] range\n")
            f.write(f"Feature Vector Size: {feature_count} features\n")
            f.write(f"  - Width: {self.img_size[0]} pixels\n")
            f.write(f"  - Height: {self.img_size[1]} pixels\n")
            f.write(f"  - Channels: 3 (RGB)\n")
            f.write(f"  - Total: {self.img_size[0]} × {self.img_size[1]} × 3 = {feature_count}\n\n")
            
            f.write("DATA QUALITY\n")
            f.write("-" * 60 + "\n")
            f.write("Outlier Removal: Yes\n")
            f.write("  - Criteria: Brightness and contrast thresholds\n")
            f.write("  - Mean brightness: 0.05 - 0.95\n")
            f.write("  - Standard deviation: > 0.02\n")
            f.write("Class Balancing: Stratified split\n")
            f.write("Random Seed: 42 (for reproducibility)\n\n")
            
            f.write("CSV FILE STRUCTURE\n")
            f.write("-" * 60 + "\n")
            f.write("Columns:\n")
            f.write("  - filename: Original image filename\n")
            f.write("  - flower_type: Flower category name\n")
            f.write("  - label: Numeric label (0-4)\n")
            f.write(f"  - pixel_0 to pixel_{feature_count-1}: Normalized pixel values\n\n")
            
            f.write("DATA SOURCE\n")
            f.write("-" * 60 + "\n")
            f.write("Source: Web scraping from GitHub repository\n")
            f.write("Method: GitHub API\n")
            f.write("Purpose: Educational AI/ML course project\n")
            f.write("Course: Advanced AI-based Application Systems\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("END OF METADATA\n")
            f.write("=" * 60 + "\n")
        
        print(f"  ✓ Metadata saved: {metadata_path}")
    
    def run_preparation(self):
        """Execute full data preparation pipeline"""
        print("\n" + "=" * 60)
        print("🚀 FLOWER DATA PREPARATION PIPELINE")
        print("=" * 60)
        print(f"\nInput Directory: {self.data_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Image Size: {self.img_size[0]}x{self.img_size[1]}")
        print(f"Flower Types: {', '.join(self.flower_types)}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Step 1: Load and process images
        all_data = self.load_and_process_images()
        
        if not all_data:
            print("\n✗ ERROR: No data found! Please check:")
            print("  1. Images are in data/flowers/ folder")
            print("  2. Folder names are correct (sunflower, daisy, etc.)")
            print("  3. Files are valid images (.jpg, .png, etc.)")
            return
        
        # Step 2: Remove outliers
        cleaned_data = self.remove_outliers(all_data)
        
        if len(cleaned_data) < 10:
            print("\n⚠ WARNING: Very few images remaining after cleaning!")
            print("  Consider adjusting outlier removal thresholds")
        
        # Step 3: Create CSV files
        train_df, test_df, activation_df = self.create_csv_files(cleaned_data)
        
        if train_df is None:
            return
        
        # Step 4: Generate metadata
        self.generate_metadata(train_df, test_df)
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ DATA PREPARATION COMPLETE!")
        print("=" * 60)
        print("\n📂 Generated Files:")
        print("  ✓ joint_data_collection.csv")
        print("  ✓ training_data.csv (80%)")
        print("  ✓ test_data.csv (20%)")
        print("  ✓ activation_data.csv (5 samples)")
        print("  ✓ dataset_metadata.txt")
        
        print("\n📊 Quick Stats:")
        print(f"  - Total images: {len(cleaned_data)}")
        print(f"  - Training: {len(train_df)}")
        print(f"  - Testing: {len(test_df)}")
        print(f"  - Features per image: {self.img_size[0] * self.img_size[1] * 3}")
        
        print("\n🎯 Next Steps:")
        print("  1. Review CSV files in data/ folder")
        print("  2. Read dataset_metadata.txt for details")
        print("  3. Proceed with ANN model training")
        print("\n" + "=" * 60)


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("  FLOWER DATA PREPARATION - SETUP")
    print("=" * 60)
    
    # Get user confirmation
    print("\n📋 Configuration:")
    print("  Input: data/flowers/")
    print("  Output: data/")
    print("  Image size: 64x64 pixels")
    print("  Split: 80% train, 20% test")
    
    confirm = input("\nProceed with data preparation? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Operation cancelled.")
        return
    
    # Create preparator and run
    preparator = FlowerDataPreparation(
        data_dir='data/flowers',
        output_dir='data'
    )
    
    preparator.run_preparation()


if __name__ == "__main__":
    main()