# German Credit Risk Classification

**Course**: M. Grum: Advanced AI-based Application Systems  
**Institution**: Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam  
**Project Type**: AI-based Credit Risk Assessment System  
**Dataset**: German Credit Risk Dataset

---

## 📋 Project Overview

This project implements an AI-based system for classifying credit risk using the German Credit Risk dataset. The system predicts whether a person represents a good or bad credit risk based on various financial and personal attributes.

The project includes:
- Data scraping and preprocessing
- Artificial Neural Network (ANN) model
- Ordinary Least Squares (OLS) regression model
- Comprehensive model comparison and evaluation
- Docker-based deployment infrastructure

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
│   ├── 02_data_preparation.py       # Data cleaning and preparation
│   ├── 03_ann_model.py              # ANN model training
│   └── 04_ols_model.py              # OLS model training and comparison
├── data/
│   ├── german_credit_raw.csv        # Raw scraped data
│   ├── joint_data_collection.csv    # Cleaned and normalized data
│   ├── training_data.csv            # 80% training set
│   ├── test_data.csv                # 20% test set
│   └── activation_data.csv          # Single sample for testing
├── images/
│   ├── learningBase/                # Training/test data and ANN model
│   │   ├── data/                    # Training and test datasets
│   │   ├── currentAiSolution.h5     # Trained ANN model
│   │   ├── training_metrics.json    # ANN performance metrics
│   │   ├── diagnostic_plots.png     # ANN diagnostic visualizations
│   │   └── training_report.txt      # ANN training report
│   ├── activationBase/              # Activation data
│   │   └── data/
│   │       └── activation_data.csv  # Single test sample
│   ├── olsBase/                     # OLS model and comparison
│   │   ├── currentOlsSolution.pkl   # Trained OLS model (pickle)
│   │   ├── currentOlsSolution.xml   # Trained OLS model (XML)
│   │   ├── ols_metrics.json         # OLS performance metrics
│   │   ├── ols_diagnostic_scatter.png           # OLS diagnostics
│   │   ├── ols_vs_ann_comparison.png            # Model comparison chart
│   │   ├── ols_vs_ann_detailed_comparison.png   # Detailed comparison
│   │   └── ols_training_report.txt  # OLS training report
│   ├── knowledgeBase/               # Docker image for AI/OLS models
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── docker-compose.yml
│   │   ├── currentAiSolution.h5
│   │   ├── currentOlsSolution.pkl
│   │   └── currentOlsSolution.xml
│   └── codeBase/                    # Docker image for activation data
│       ├── Dockerfile
│       ├── README.md
│       ├── docker-compose.yml
│       └── activation_data.csv
├── docker-compose.yml               # Unified Docker Compose configuration
└── README.md
```

---

## 🐳 Docker Images

This project provides four Docker images for data and model management:

### 1. Learning Base Image
Contains training and test data for model development.

**Pull command:**
```bash
docker pull devangthaker/learningbase_germancreditrisk:latest
```

**Data locations:**
- Training data: `/tmp/learningBase/train/training_data.csv`
- Test data: `/tmp/learningBase/validation/test_data.csv`

### 2. Knowledge Base Image
Contains trained AI and OLS models.

**Pull command:**
```bash
docker pull devangthaker/knowledgebase_germancreditrisk:latest
```

**Model locations:**
- ANN model: `/tmp/knowledgeBase/currentAiSolution.h5`
- OLS model (pickle): `/tmp/knowledgeBase/currentOlsSolution.pkl`
- OLS model (XML): `/tmp/knowledgeBase/currentOlsSolution.xml`

### 3. Code Base Image
Contains activation data for model testing.

**Pull command:**
```bash
docker pull devangthaker/codebase_germancreditrisk:latest
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

3. **Start all services**
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

### Accessing Data and Models

**From container shell:**
```bash
# Access Learning Base
docker-compose exec learningbase sh

# Access Knowledge Base
docker-compose exec knowledgebase sh

# Access Code Base
docker-compose exec codebase sh
```

**View data files:**
```bash
# Training data
docker-compose exec learningbase ls -lh /tmp/learningBase/train/

# Test data
docker-compose exec learningbase ls -lh /tmp/learningBase/validation/

# AI/OLS models
docker-compose exec knowledgebase ls -lh /tmp/knowledgeBase/

# Activation data
docker-compose exec codebase ls -lh /tmp/activationBase/
```

---

## 🤖 Models

### Artificial Neural Network (ANN)

**Architecture:**
- Input Layer: 9 features
- Hidden Layer 1: 128 neurons (ReLU) + Batch Normalization + Dropout (0.3)
- Hidden Layer 2: 64 neurons (ReLU) + Batch Normalization + Dropout (0.3)
- Hidden Layer 3: 32 neurons (ReLU) + Batch Normalization + Dropout (0.3)
- Hidden Layer 4: 16 neurons (ReLU) + Dropout (0.3)
- Output Layer: 1 neuron (Sigmoid)

**Training Configuration:**
- Optimizer: Adam (learning_rate=0.001)
- Loss Function: Binary Crossentropy
- Batch Size: 32
- Epochs: 55

**Performance Metrics:**
- Test Accuracy: 76.36%
- Test Precision: 80.90%
- Test Recall: 88.89%
- Test F1-Score: 84.71%

### Ordinary Least Squares (OLS)

**Model Type:** Linear Regression using Statsmodels

**Performance Metrics:**
- Test Accuracy: 74.55%
- Test Precision: 76.82%
- Test Recall: 93.77%
- Test F1-Score: 84.43%

### Model Comparison

Both models demonstrate strong performance, with the ANN slightly outperforming OLS in accuracy and precision, while OLS shows higher recall. Detailed comparison visualizations are available in the `olsBase` directory.

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

### Subgoal 3: Docker Images for Data ✅
- Two Docker images created (learningBase, activationBase)
- Images based on busybox
- Published to Docker Hub
- README.md included in each image with:
  - Ownership information
  - Course and institution details
  - Data source attribution
  - AGPL-3.0 license statement
- docker-compose.yml using external volume `ai_system`
- Data accessible at specified paths

### Subgoal 4: AI Model Development ✅
- Deep Neural Network (ANN) implemented using TensorFlow/Keras
- Model architecture: 4 hidden layers with batch normalization and dropout
- Training performed on prepared dataset
- Model saved as `currentAiSolution.h5`
- Performance metrics documented:
  - Training accuracy, validation accuracy
  - Test accuracy, precision, recall, F1-score
- Visualizations generated:
  - Training/testing curves (loss and accuracy)
  - Diagnostic plots (confusion matrix, ROC curve, PR curve)
  - Scatter plots (actual vs predicted)
- Training report generated with comprehensive documentation

### Subgoal 5: OLS Model and Comparison ✅
- OLS regression model implemented using Statsmodels
- Model trained on same dataset as ANN
- Model saved in multiple formats:
  - `currentOlsSolution.pkl` (pickle format)
  - `currentOlsSolution.xml` (XML format)
- Testing routines implemented for validation
- Performance metrics calculated and stored
- Diagnostic visualizations generated:
  - Residual plots
  - Scatter plots (actual vs predicted)
- Comprehensive comparison with ANN model:
  - Side-by-side performance metrics
  - Bar chart comparisons
  - Detailed comparison dashboard
- Training report with comparison analysis

### Subgoal 6: Model Docker Provision ✅
- Two additional Docker images created:
  
  **Knowledge Base Image** (`knowledgebase_germancreditrisk`):
  - Contains trained AI and OLS models
  - Files: currentAiSolution.h5, currentOlsSolution.pkl, currentOlsSolution.xml
  - Path: `/tmp/knowledgeBase/`
  - Based on busybox image
  - Includes README.md with ownership, course info, model characterization, and AGPL-3.0 license
  
  **Code Base Image** (`codebase_germancreditrisk`):
  - Contains activation data for testing
  - File: activation_data.csv
  - Path: `/tmp/activationBase/`
  - Based on busybox image
  - Includes README.md with ownership, course info, data characterization, and AGPL-3.0 license

- Each image tested with individual docker-compose.yml
- Images published to Docker Hub
- Unified docker-compose.yml created for all services
- External volume `ai_system` used for data mounting

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

---

## 📚 References

- TensorFlow/Keras Documentation
- Statsmodels Documentation
- Docker Documentation
- Scikit-learn Documentation