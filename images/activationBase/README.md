# Flower Classification - Activation Base Image

## Ownership
**Created by:** [Your Name] & [Partner Name]  
**University:** University of Potsdam  
**Department:** Junior Chair for Business Information Science, esp. AI-based Application Systems

## Purpose
This Docker image is part of the course **"M. Grum: Advanced AI-based Application Systems"** and contains the activation dataset for testing the Flower Classification AI model.

## Data Origin
The data was scraped from the following sources using web scraping techniques:
- GitHub Repository: https://github.com/devang1010/flower-images-dataset
- Method: GitHub REST API
- Flower Types: Sunflower, Daisy, Dandelion, Rose, Tulip

## Contents
This image contains:
- `/tmp/activationBase/activation_data.csv` - Activation dataset (1 sample per flower class)

## Dataset Details
- **Image Size:** 64x64 pixels (RGB)
- **Features per image:** 12,288 (64 × 64 × 3)
- **Samples:** 5 (one per class)
- **Format:** CSV with normalized pixel values [0, 1]

## License
This project is committed to the **AGPL-3.0 license**.

## Usage
```bash
docker pull YOUR_USERNAME/activationbase_flower-classification:latest
docker run --rm activationbase_flower-classification ls -la /tmp/activationBase/
```

## Project
Part of the Flower Classification AI project using Artificial Neural Networks (ANN).