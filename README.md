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
- Docker Compose orchestration for AI and OLS model applications

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
│   ├── activationBase/              # Docker image for activation data
│   │   ├── data/
│   │   │   └── activation_data.csv
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   └── docker-compose.yml
│   ├── codeBase/                    # Docker image for activation data (deployment)
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── docker-compose.yml
│   │   └── activation_data.csv
│   ├── knowledgeBase/               # Docker image for AI/OLS models
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── docker-compose.yml
│   │   ├── currentAiSolution.h5
│   │   ├── currentOlsSolution.pkl
│   │   └── currentOlsSolution.xml
│   ├── learningBase/                # Docker image for training/test data
│   │   ├── data/
│   │   │   ├── train/
│   │   │   │   └── training_data.csv
│   │   │   └── validation/
│   │   │       └── test_data.csv
│   │   ├── currentAiSolution.h5     # Trained ANN model
│   │   ├── training_metrics.json    # ANN performance metrics
│   │   ├── diagnostic_plots.png     # ANN diagnostic visualizations
│   │   ├── training_report.txt      # ANN training report
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   └── docker-compose.yml
│   └── olsBase/                     # OLS model and comparison
│       ├── currentOlsSolution.pkl   # Trained OLS model (pickle)
│       ├── currentOlsSolution.xml   # Trained OLS model (XML)
│       ├── ols_metrics.json         # OLS performance metrics
│       ├── ols_diagnostic_scatter.png           # OLS diagnostics
│       ├── ols_vs_ann_comparison.png            # Model comparison chart
│       ├── ols_vs_ann_detailed_comparison.png   # Detailed comparison
│       └── ols_training_report.txt  # OLS training report
├── docker-compose-ai.yml            # Docker Compose for AI model application
├── docker-compose-ols.yml           # Docker Compose for OLS model application
└── README.md
```

---

## 🐳 Docker Images

This project provides three Docker images published to Docker Hub for complete model deployment:

### 1. Learning Base Image
Contains training and test data for model development.

**Pull command:**
```bash
docker pull devangthaker/learningbase_germancreditrisk:latest
```

**Image size:** ~6.96 MB

**Data locations:**
- Training data: `/tmp/learningBase/train/training_data.csv`
- Test data: `/tmp/learningBase/validation/test_data.csv`

### 2. Knowledge Base Image
Contains trained AI and OLS models.

**Pull command:**
```bash
docker pull devangthaker/knowledgebase_germancreditrisk:latest
```

**Image size:** ~7.41 MB

**Model locations:**
- ANN model: `/tmp/knowledgeBase/currentAiSolution.h5` (~224 KB)
- OLS model (pickle): `/tmp/knowledgeBase/currentOlsSolution.pkl` (~121 KB)
- OLS model (XML): `/tmp/knowledgeBase/currentOlsSolution.xml` (~688 bytes)

### 3. Code Base Image
Contains activation data for model testing.

**Pull command:**
```bash
docker pull devangthaker/codebase_germancreditrisk:latest
```

**Image size:** ~6.82 MB

**Data location:**
- Activation data: `/tmp/activationBase/activation_data.csv` (342 bytes)

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

3. **Run AI Model Application**
```bash
# Start AI composition
docker-compose -f docker-compose-ai.yml up -d

# Check running containers
docker ps

# View logs
docker-compose -f docker-compose-ai.yml logs

# Verify AI model is accessible
docker exec ai_knowledgebase ls -lh /shared_data/knowledgeBase/currentAiSolution.h5

# Stop AI composition
docker-compose -f docker-compose-ai.yml down
```

4. **Run OLS Model Application**
```bash
# Clear volume
docker volume rm ai_system
docker volume create ai_system

# Start OLS composition
docker-compose -f docker-compose-ols.yml up -d

# Check running containers
docker ps

# View logs
docker-compose -f docker-compose-ols.yml logs

# Verify OLS models are accessible
docker exec ols_knowledgebase ls -lh /shared_data/knowledgeBase/currentOlsSolution.pkl
docker exec ols_knowledgebase ls -lh /shared_data/knowledgeBase/currentOlsSolution.xml

# Stop OLS composition
docker-compose -f docker-compose-ols.yml down
```

### Docker Compose Files

#### AI Model Composition (`docker-compose-ai.yml`)
- Deploys the Artificial Neural Network model
- Uses all three Docker Hub images
- Mounts external volume `ai_system` to `/shared_data`
- Clears existing volume content on startup
- Provides access to:
  - Training/test data
  - `currentAiSolution.h5` (Keras model)
  - Activation data for testing

#### OLS Model Composition (`docker-compose-ols.yml`)
- Deploys the Ordinary Least Squares regression model
- Uses all three Docker Hub images
- Mounts external volume `ai_system` to `/shared_data`
- Clears existing volume content on startup
- Provides access to:
  - Training/test data
  - `currentOlsSolution.pkl` (Pickle format)
  - `currentOlsSolution.xml` (XML format)
  - Activation data for testing

### Accessing Data and Models

**Verify AI model files:**
```bash
docker exec ai_knowledgebase ls -lh /shared_data/knowledgeBase/
docker exec ai_codebase ls -lh /shared_data/activationBase/
docker exec ai_learningbase ls -lh /shared_data/learningBase/
```

**Verify OLS model files:**
```bash
docker exec ols_knowledgebase ls -lh /shared_data/knowledgeBase/
docker exec ols_codebase ls -lh /shared_data/activationBase/
docker exec ols_learningbase ls -lh /shared_data/learningBase/
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

**Deployment:**
- Model file: `currentAiSolution.h5` (~224 KB)
- Format: Keras HDF5
- Accessed via: `docker-compose-ai.yml`

### Ordinary Least Squares (OLS)

**Model Type:** Linear Regression using Statsmodels

**Performance Metrics:**
- Test Accuracy: 74.55%
- Test Precision: 76.82%
- Test Recall: 93.77%
- Test F1-Score: 84.43%

**Deployment:**
- Model files:
  - `currentOlsSolution.pkl` (~121 KB) - Scikit-learn pickle format
  - `currentOlsSolution.xml` (~688 bytes) - XML metadata
- Accessed via: `docker-compose-ols.yml`

### Model Comparison

Both models demonstrate strong performance, with the ANN slightly outperforming OLS in accuracy and precision, while OLS shows higher recall. The Docker Compose infrastructure allows easy deployment and switching between both models using the same data pipeline.

**Key Differences:**

| Aspect | AI Model | OLS Model |
|--------|----------|-----------|
| Type | Deep Neural Network | Linear Regression |
| File Format | HDF5 (.h5) | Pickle (.pkl) + XML |
| File Size | ~224 KB | ~121 KB + ~688 bytes |
| Accuracy | 76.36% | 74.55% |
| Precision | 80.90% | 76.82% |
| Recall | 88.89% | 93.77% |
| F1-Score | 84.71% | 84.43% |

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

### Subgoal 7: Docker Builds and Docker-Compose Utilization ✅

Created two separate Docker Compose configurations for deploying AI and OLS model applications:

**AI Model Application** (`docker-compose-ai.yml`):
- Uses three Docker Hub images:
  - `devangthaker/learningbase_germancreditrisk:latest`
  - `devangthaker/knowledgebase_germancreditrisk:latest`
  - `devangthaker/codebase_germancreditrisk:latest`
- Mounts external volume `ai_system:/shared_data`
- Clears `/shared_data/*` content on startup
- Provides access to:
  - Training/validation data from learningBase
  - AI model (`currentAiSolution.h5`) from knowledgeBase
  - Activation data from codeBase
- Container names: `ai_learningbase`, `ai_knowledgebase`, `ai_codebase`
- Network: `ai-cps_ai_network`

**OLS Model Application** (`docker-compose-ols.yml`):
- Uses same three Docker Hub images
- Mounts external volume `ai_system:/shared_data`
- Clears `/shared_data/*` content on startup
- Provides access to:
  - Training/validation data from learningBase
  - OLS models (`currentOlsSolution.pkl`, `currentOlsSolution.xml`) from knowledgeBase
  - Activation data from codeBase
- Container names: `ols_learningbase`, `ols_knowledgebase`, `ols_codebase`
- Network: `ai-cps_ols_network`

**Key Features:**
- Both compositions use the same external volume for data persistence
- Volume content cleared before each deployment to ensure clean state
- All containers remain running with `tail -f /dev/null` for data access
- Comprehensive logging showing successful data copying and file verification
- Models and data accessible via shared volume across containers

**Verification:**
- AI model file accessible at ~224 KB
- OLS model files accessible at ~121 KB (pkl) and ~688 bytes (xml)
- Both compositions successfully tested and verified
- Clean separation between AI and OLS deployments

---

## 🧪 Testing

### Automated Verification

The test script verifies:
- External volume creation
- Container startup for both compositions
- File accessibility (AI and OLS models)
- Activation data availability
- Log message correctness
- Clean shutdown of services

### Manual Testing

**Test AI Composition:**
```bash
docker volume create ai_system
docker-compose -f docker-compose-ai.yml up -d
docker exec ai_knowledgebase ls -lh /shared_data/knowledgeBase/currentAiSolution.h5
docker-compose -f docker-compose-ai.yml logs
docker-compose -f docker-compose-ai.yml down
```

**Test OLS Composition:**
```bash
docker volume rm ai_system && docker volume create ai_system
docker-compose -f docker-compose-ols.yml up -d
docker exec ols_knowledgebase ls -lh /shared_data/knowledgeBase/currentOlsSolution.pkl
docker exec ols_knowledgebase ls -lh /shared_data/knowledgeBase/currentOlsSolution.xml
docker-compose -f docker-compose-ols.yml logs
docker-compose -f docker-compose-ols.yml down
```

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

- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Statsmodels Documentation: https://www.statsmodels.org/
- Docker Documentation: https://docs.docker.com/
- Docker Compose Documentation: https://docs.docker.com/compose/
- Scikit-learn Documentation: https://scikit-learn.org/
- Docker Hub: https://hub.docker.com/

---

## 🎯 Project Status

**Current Status:** ✅ All Subgoals Complete (1-7)

All project requirements have been successfully implemented, tested, and verified. Both AI and OLS model applications are fully deployable using Docker Compose with proper volume management and data accessibility.