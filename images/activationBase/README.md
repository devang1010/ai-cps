# Activation Base - German Credit Risk AI Project

## Ownership
**Authors**: Devang Thaker & Krish Manvar
**Course**: M. Grum: Advanced AI-based Application Systems
**Institution**: Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam
**Date**: January 2026

## Purpose
This Docker image provides a single test sample for activating and testing the German Credit Risk prediction AI system.

## Data Organization
- **Activation Data**: `/tmp/activationBase/activation_data.csv` (1 sample from test set)

## Data Source
- **Original Dataset**: German Credit Data from GitHub repository
- **Repository**: https://github.com/devang1010/German-credit-score
- **File**: german_credit_data.csv
- **Sample**: Randomly selected from test_data.csv

## License
This project is licensed under the **AGPL-3.0 License**.

## Data Description
This image contains a single credit application record that can be used to:
- Test the trained AI model
- Demonstrate prediction capabilities
- Validate model deployment
- Perform single-instance inference

## Usage

### Pull the image
```bash
docker pull [your-dockerhub-username]/activationbase_germancreditrisk:latest
```

### Run the container
```bash
docker run --rm [your-dockerhub-username]/activationbase_germancreditrisk:latest
```

### Use with docker-compose
```bash
docker-compose up
```

### Access data in volumes
The image mounts data to `/tmp` which can be accessed via external volumes.

## Image Details
- **Base Image**: busybox:latest
- **Size**: Minimal (single data entry)
- **Architecture**: Multi-platform support

## Contact
For questions or issues, please contact through the University of Potsdam course portal.