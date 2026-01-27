# German Credit Risk Classification

**Course**: M. Grum: Advanced AI-based Application Systems  
**Institution**: Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam  
**Project Type**: AI-based Credit Risk Assessment System  
**Dataset**: German Credit Risk Dataset

---

## 📋 Project Overview

This project implements an AI-based system for classifying credit risk using the German Credit Risk dataset. The system predicts whether a person represents a good or bad credit risk based on various financial and personal attributes.

This repository is forked from [MarcusGrum/AI-CPS](https://github.com/MarcusGrum/AI-CPS) as part of the course requirements.

---

## 👥 Team Members

- **Devang Thaker**
- **Krish Manvar**

---

## 📊 Dataset Information

### Source
- **Original Dataset**: German Credit Data from GitHub repository
- **Repository**: https://github.com/devang1010/German-credit-score
- **File**: german_credit_data.csv

### Attributes

| Attribute | Type | Description | Values/Range |
|-----------|------|-------------|--------------|
| Age | Numeric | Age of the person | Years |
| Sex | Categorical | Gender | male, female |
| Job | Numeric | Employment status | 0: unskilled & non-resident<br>1: unskilled & resident<br>2: skilled<br>3: highly skilled |
| Housing | Categorical | Housing situation | own, rent, free |
| Saving accounts | Categorical | Savings level | little, moderate, quite rich, rich |
| Checking account | Numeric | Checking account balance | DM (Deutsche Mark) |
| Credit amount | Numeric | Loan amount requested | DM |
| Duration | Numeric | Loan duration | Months |
| Purpose | Categorical | Purpose of loan | car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others |

### Target Variable
- **Risk Classification**: Good or Bad credit risk

---

## 🚀 Project Structure
```
AI-CPS/
├── code/
│   ├── 01_data_scrapper.py          # Web scraping script
│   └── 02_data_preparation.py       # Data cleaning and preparation
├── data/
│   ├── german_credit_raw.csv        # Raw scraped data
│   ├── joint_data_collection.csv    # Cleaned and normalized data
│   ├── training_data.csv            # 80% training set
│   ├── test_data.csv                # 20% test set
│   └── activation_data.csv          # Single sample for testing
├── images/
│   ├── learningBase/                # Docker image for training/test data
│   └── activationBase/              # Docker image for activation data
├── docker-compose.yml               # Unified Docker Compose configuration
└── README.md
```

---

## 🐳 Docker Images

This project provides two Docker images for data management:

### Learning Base Image
Contains training and test data for model development.

**Pull command:**
```bash
docker pull devangthaker/learningbase_germancreditrisk:latest
```

**Data locations:**
- Training data: `/tmp/learningBase/train/training_data.csv`
- Test data: `/tmp/learningBase/validation/test_data.csv`

### Activation Base Image
Contains a single test sample for model activation and testing.

**Pull command:**
```bash
docker pull devangthaker/activationbase_germancreditrisk:latest
```

**Data location:**
- Activation data: `/tmp/activationBase/activation_data.csv`

---

## 🔧 Usage

### Prerequisites
- Docker installed and running
- Docker Compose installed

### Quick Start

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd AI-CPS
```

2. **Create external volume**
```bash
docker volume create ai_system
```

3. **Start services**
```bash
docker-compose up -d
```

4. **Check status**
```bash
docker-compose ps
```

5. **View logs**
```bash
docker-compose logs
```

6. **Stop services**
```bash
docker-compose down
```

### Accessing Data

**From container shell:**
```bash
# Access Learning Base
docker-compose exec learningbase sh

# Access Activation Base
docker-compose exec activationbase sh
```

**View data files:**
```bash
# Training data
docker-compose exec learningbase ls -lh /tmp/learningBase/train/

# Test data
docker-compose exec learningbase ls -lh /tmp/learningBase/validation/

# Activation data
docker-compose exec activationbase cat /tmp/activationBase/activation_data.csv
```

---

## 📝 Project Workflow

### Subgoal 1: Git Usage ✅
- Repository forked from MarcusGrum/AI-CPS
- Project structure established
- Multiple commits from both team members
- All changes pushed to GitHub

### Subgoal 2: Data Scraping and Preparation ✅
- Data scraped from GitHub repository
- Data cleaned (missing values, outliers, normalization)
- Data split into training (80%) and test (20%) sets
- Activation data created (single test sample)
- Files generated:
  - `joint_data_collection.csv`
  - `training_data.csv`
  - `test_data.csv`
  - `activation_data.csv`

### Subgoal 3: Docker Images ✅
- Two Docker images created based on busybox
- Images published to Docker Hub
- README.md included in each image with:
  - Ownership information
  - Course and institution details
  - Data source attribution
  - AGPL-3.0 license statement
- docker-compose.yml using external volume `ai_system`
- Data accessible at specified paths

---

## 📄 License

This project is licensed under the **AGPL-3.0 License**.

---

## 📧 Contact

For questions or issues, please contact through the University of Potsdam course portal.

---

## 🙏 Acknowledgments

- **Course**: M. Grum: Advanced AI-based Application Systems
- **Institution**: Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam
- **Original Repository**: [MarcusGrum/AI-CPS](https://github.com/MarcusGrum/AI-CPS)
- **Dataset Source**: [German Credit Score Repository](https://github.com/devang1010/German-credit-score)