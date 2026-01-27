# Learning Base - German Credit Risk AI Project

## Ownership
**Authors**: Devang Thaker & Krish Manvar
**Course**: M. Grum: Advanced AI-based Application Systems
**Institution**: Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam
**Date**: January 2026

## Purpose
This Docker image provides training and test data for the German Credit Risk prediction AI system.

## Data Organization
- **Training Data**: `/tmp/learningBase/train/training_data.csv` (80% of dataset)
- **Validation Data**: `/tmp/learningBase/validation/test_data.csv` (20% of dataset)

## Data Source
- **Original Dataset**: German Credit Data from GitHub repository
- **Repository**: https://github.com/devang1010/German-credit-score
- **File**: german_credit_data.csv

## License
This project is licensed under the **AGPL-3.0 License**.

## Data Description
The German Credit dataset contains information about credit applicants and their credit risk classification. The data includes:
- Demographic information
- Financial history
- Credit characteristics
- Risk assessment (target variable)

## Usage

### Pull the image
```bash
docker pull [your-dockerhub-username]/learningbase_germancreditrisk:latest
```

### Run the container
```bash
docker run --rm [your-dockerhub-username]/learningbase_germancreditrisk:latest
```

### Use with docker-compose
```bash
docker-compose up
```

### Access data in volumes
The image mounts data to `/tmp` which can be accessed via external volumes.

## Image Details
- **Base Image**: busybox:latest
- **Size**: Minimal (optimized for data storage)
- **Architecture**: Multi-platform support

## Contact
For questions or issues, please contact through the University of Potsdam course portal.