# Flower Classification - Learning Base Image

## Ownership
**Created by:** [Your Name] & [Partner Name]  
**University:** University of Potsdam  
**Department:** Junior Chair for Business Information Science, esp. AI-based Application Systems

## Purpose
This Docker image is part of the course **"M. Grum: Advanced AI-based Application Systems"** and contains the training and validation datasets for the Flower Classification AI project.

## Data Origin
The data was scraped from the following sources using web scraping techniques:
- GitHub Repository: https://github.com/devang1010/flower-images-dataset
- Method: GitHub REST API
- Flower Types: Sunflower, Daisy, Dandelion, Rose, Tulip
- Total Images: ~250 images processed and converted to CSV format

## Contents
This image contains:
- `/tmp/learningBase/train/training_data.csv` - Training dataset (80% of total data)
- `/tmp/learningBase/validation/test_data.csv` - Validation/test dataset (20% of total data)

## Dataset Details
- **Image Size:** 64x64 pixels (RGB)
- **Features per image:** 12,288 (64 × 64 × 3)
- **Classes:** 5 flower types
- **Format:** CSV with normalized pixel values [0, 1]

## License
This project is committed to the **AGPL-3.0 license**.

## Usage
```bash
docker pull YOUR_USERNAME/learningbase_flower-classification:latest
docker run --rm learningbase_flower-classification ls -la /tmp/learningBase/
```

## Project
Part of the Flower Classification AI project using Artificial Neural Networks (ANN).