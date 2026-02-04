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
- Docker-based deployment infrastructure with **automatic predictions**
- Separate Docker Compose files for ANN and OLS model deployments

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
│   ├── codeBase/                    # Docker image for predictions
│   │   ├── Dockerfile               # Python 3.12 + TensorFlow + Statsmodels
│   │   ├── README.md
│   │   ├── activation_data.csv
│   │   ├── predict_ann.py           # ANN prediction script
│   │   └── predict_ols.py           # OLS prediction script
│   ├── knowledgeBase/               # Docker image for AI/OLS models
│   │   ├── Dockerfile
│   │   ├── README.md
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
│   │   └── README.md
│   └── olsBase/                     # OLS model and comparison
│       ├── currentOlsSolution.pkl   # Trained OLS model (pickle)
│       ├── currentOlsSolution.xml   # Trained OLS model (XML)
│       ├── ols_metrics.json         # OLS performance metrics
│       ├── ols_diagnostic_scatter.png           # OLS diagnostics
│       ├── ols_vs_ann_comparison.png            # Model comparison chart
│       ├── ols_vs_ann_detailed_comparison.png   # Detailed comparison
│       └── ols_training_report.txt  # OLS training report
├── scenarios/
│   ├── aiDockerImage/
│   │   └── docker-compose.yml       # Docker Compose for ANN predictions
│   └── olsDockerImage/
│       └── docker-compose.yml       # Docker Compose for OLS predictions
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

### 3. Code Base Image (Prediction Engine)
Contains activation data and prediction scripts for both ANN and OLS models.

**Pull command:**
```bash
docker pull devangthaker/codebase_germancreditrisk:latest
```

**Image details:**
- Base image: `python:3.12-slim`
- Includes: TensorFlow 2.20.0, Keras 3.13.1, Statsmodels 0.14.6, Pandas 3.0.0, NumPy 2.4.1
- Scripts: `predict_ann.py`, `predict_ols.py`
- Data: `activation_data.csv`

**Features:**
- ✅ Automatic model loading
- ✅ Automatic predictions on startup
- ✅ Beautiful formatted output with emojis
- ✅ Confidence scores and probabilities
- ✅ Error handling and diagnostics

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

2. **Create external volume (one-time setup)**
```bash
docker volume create ai_system
```

3. **Run ANN Model with Automatic Predictions**
```bash
cd scenarios/aiDockerImage
docker-compose up
```

**Expected Output:**
```
✅ Learning Base: Data copied to shared volume
✅ Knowledge Base: Models copied to shared volume
✅ Code Base: Activation data copied
⏳ Waiting for models to be ready...

======================================================================
🤖 ANN MODEL PREDICTION
======================================================================

📥 Loading ANN model...
✅ Model loaded successfully!

📊 Loading activation data...
✅ Loaded 1 sample(s)

📋 Features used: ['Age', 'Job', 'Credit amount', 'Duration', ...]
📏 Number of features: 9

🔮 Making predictions...

======================================================================
📈 PREDICTION RESULTS
======================================================================

Sample 1:
  Prediction: ✅ GOOD CREDIT
  Probability: 0.8234
  Confidence: 82.34%
  Classification: Good

======================================================================
✅ ANN Prediction Complete!
======================================================================
```

Press `Ctrl+C` to stop, then:
```bash
docker-compose down
```

4. **Run OLS Model with Automatic Predictions**
```bash
cd scenarios/olsDockerImage
docker-compose up
```

**Expected Output:**
```
✅ Learning Base: Data copied to shared volume
✅ Knowledge Base: Models copied to shared volume
✅ Code Base: Activation data copied
⏳ Waiting for models to be ready...

======================================================================
📊 OLS MODEL PREDICTION
======================================================================

📥 Loading OLS model...
✅ Model loaded successfully!

📊 Loading activation data...
✅ Loaded 1 sample(s)

📋 Features used: ['Age', 'Job', 'Credit amount', 'Duration', ...]
📏 Number of features: 9

🔮 Making predictions...

======================================================================
📈 PREDICTION RESULTS
======================================================================

Sample 1:
  Prediction: ✅ GOOD CREDIT
  Probability: 0.7891
  Confidence: 78.91%
  Classification: Good

======================================================================
✅ OLS Prediction Complete!
======================================================================
```

Press `Ctrl+C` to stop, then:
```bash
docker-compose down
```

### Docker Compose Architecture

Both compose files use the same architecture with automatic prediction execution:

```yaml
services:
  learningbase:   # Copies training/test data to shared volume
  knowledgebase:  # Copies trained models to shared volume
  codebase:       # Copies activation data, then runs predictions automatically
```

**Key Features:**
- 🔄 Automatic data synchronization via shared volume
- 🚀 Predictions run automatically on container startup
- 📊 Real-time output in terminal
- ✅ Models and data verified before predictions
- 🎯 Clean separation between ANN and OLS deployments

### Manual Prediction Execution

If you want to run predictions manually after containers are running:

```bash
# Start containers in detached mode
docker-compose up -d

# Run ANN prediction manually
docker exec ann_codebase python /scripts/predict_ann.py

# Or run OLS prediction manually
docker exec ols_codebase python /scripts/predict_ols.py

# View logs
docker logs ann_codebase
docker logs ols_codebase

# Stop containers
docker-compose down
```

### Accessing Data and Models

**Verify ANN files:**
```bash
docker exec ann_knowledgebase ls -lh /shared_data/knowledgeBase/
docker exec ann_codebase ls -lh /shared_data/activationBase/
docker exec ann_learningbase ls -lh /shared_data/learningBase/
```

**Verify OLS files:**
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
- Prediction: Automatic via `predict_ann.py`
- Accessed via: `scenarios/aiDockerImage/docker-compose.yml`

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
- Prediction: Automatic via `predict_ols.py`
- Accessed via: `scenarios/olsDockerImage/docker-compose.yml`

### Model Comparison

Both models demonstrate strong performance, with the ANN slightly outperforming OLS in accuracy and precision, while OLS shows higher recall. The Docker infrastructure allows easy deployment and automatic prediction execution for both models.

**Key Differences:**

| Aspect | ANN Model | OLS Model |
|--------|-----------|-----------|
| Type | Deep Neural Network | Linear Regression |
| File Format | HDF5 (.h5) | Pickle (.pkl) + XML |
| File Size | ~224 KB | ~121 KB + ~688 bytes |
| Accuracy | 76.36% | 74.55% |
| Precision | 80.90% | 76.82% |
| Recall | 88.89% | 93.77% |
| F1-Score | 84.71% | 84.43% |
| Prediction Script | predict_ann.py | predict_ols.py |
| Dependencies | TensorFlow, Keras | Statsmodels, Scipy |

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
- Docker images created (learningBase, knowledgeBase, codeBase)
- Published to Docker Hub
- README.md included with ownership, course info, and AGPL-3.0 license
- External volume `ai_system` for data sharing

### Subgoal 4: AI Model Development ✅
- Deep Neural Network (ANN) implemented using TensorFlow/Keras
- Model architecture: 4 hidden layers with batch normalization and dropout
- Training performed with comprehensive metrics tracking
- Model saved as `currentAiSolution.h5`
- Visualizations generated:
  - Training/testing curves (loss and accuracy)
  - Diagnostic plots (confusion matrix, ROC curve, PR curve)
  - Scatter plots (actual vs predicted)
- Training report generated with comprehensive documentation

### Subgoal 5: OLS Model and Comparison ✅
- OLS regression model implemented using Statsmodels
- Model saved in multiple formats (pickle and XML)
- Testing routines implemented for validation
- Performance metrics calculated and stored
- Comprehensive comparison with ANN model:
  - Side-by-side performance metrics
  - Bar chart comparisons
  - Detailed comparison dashboard
- Training report with comparison analysis

### Subgoal 6: Model Docker Provision ✅
- Three Docker images created and published:
  
  **Learning Base Image**:
  - Contains training and test data
  - Path: `/tmp/learningBase/`
  
  **Knowledge Base Image**:
  - Contains trained AI and OLS models
  - Path: `/tmp/knowledgeBase/`
  
  **Code Base Image** (Enhanced):
  - Python 3.12 environment with ML libraries
  - Automatic prediction capabilities
  - Path: `/tmp/activationBase/` and `/scripts/`

- Each image tested and verified
- Images published to Docker Hub

### Subgoal 7: Docker Builds and Docker-Compose Utilization ✅

Created separate Docker Compose configurations with **automatic prediction execution**:

**ANN Model Deployment** (`scenarios/aiDockerImage/docker-compose.yml`):
- Three-container architecture (learningbase, knowledgebase, codebase)
- Uses external volume `ai_system:/shared_data`
- **Automatically runs ANN predictions on startup**
- Beautiful formatted output with predictions, probabilities, and confidence scores
- Container names: `ann_learningbase`, `ann_knowledgebase`, `ann_codebase`
- Network: `ann_network`

**OLS Model Deployment** (`scenarios/olsDockerImage/docker-compose.yml`):
- Three-container architecture (learningbase, knowledgebase, codebase)
- Uses external volume `ai_system:/shared_data`
- **Automatically runs OLS predictions on startup**
- Beautiful formatted output with predictions, probabilities, and confidence scores
- Container names: `ols_learningbase`, `ols_knowledgebase`, `ols_codebase`
- Network: `ols_network`

**Key Features:**
- ✅ Predictions execute automatically on `docker-compose up`
- ✅ No manual intervention required
- ✅ Real-time output visible in terminal
- ✅ Error handling and diagnostics built-in
- ✅ Clean separation between ANN and OLS deployments
- ✅ Professional output formatting with emojis and clear sections
- ✅ Model verification before predictions
- ✅ Both deployments use the same shared codebase image

**Innovation:**
Unlike typical Docker deployments that only store models, this implementation includes **intelligent prediction engines** that automatically:
1. Load the appropriate model (ANN or OLS)
2. Load and validate activation data
3. Execute predictions with proper error handling
4. Display results with probabilities and confidence scores
5. Provide clear visual feedback of the entire process

---

## 🧪 Testing

### Automated Verification

The Docker Compose files automatically verify:
- External volume creation and mounting
- Container startup sequence (learningbase → knowledgebase → codebase)
- File accessibility (models and data)
- Model loading success
- Prediction execution
- Output formatting

### Manual Testing

**Test ANN Composition:**
```bash
docker volume create ai_system
cd scenarios/aiDockerImage
docker-compose up
# Watch automatic predictions in terminal
# Press Ctrl+C to stop
docker-compose down
```

**Test OLS Composition:**
```bash
cd scenarios/olsDockerImage
docker-compose up
# Watch automatic predictions in terminal
# Press Ctrl+C to stop
docker-compose down
```

**Manual Prediction Testing:**
```bash
# Start in detached mode
docker-compose up -d

# Test ANN prediction
docker exec ann_codebase python /scripts/predict_ann.py

# Test OLS prediction
docker exec ols_codebase python /scripts/predict_ols.py

# View logs
docker logs ann_codebase
docker logs ols_codebase

# Cleanup
docker-compose down
```

---

## 🎯 Key Innovations

### 1. Automatic Prediction Execution
- Models automatically make predictions on container startup
- No manual script execution required
- Perfect for demonstrations and automated pipelines

### 2. Intelligent Error Handling
- Validates model and data availability
- Provides clear error messages
- Graceful failure with diagnostic information

### 3. Professional Output Formatting
- Beautiful terminal output with emojis
- Clear sections and separators
- Confidence scores and probabilities
- Easy-to-read prediction results

### 4. Unified Codebase Architecture
- Single codebase image supports both ANN and OLS
- Different compose files select which model to use
- Reduces image redundancy
- Simplifies maintenance

### 5. Production-Ready Deployment
- External volume for data persistence
- Network isolation between deployments
- Container dependency management
- Restart policies for reliability

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
- Python Documentation: https://docs.python.org/3/

---

## 🎯 Project Status

**Current Status:** ✅ All Subgoals Complete (1-7) + Automatic Prediction System

All project requirements have been successfully implemented, tested, and verified. The system now includes:
- ✅ Complete data pipeline (scraping, preparation, splitting)
- ✅ Two trained models (ANN and OLS)
- ✅ Three Docker images published to Docker Hub
- ✅ Separate Docker Compose files for ANN and OLS deployments
- ✅ **Automatic prediction execution on container startup**
- ✅ Beautiful formatted output with confidence scores
- ✅ Professional error handling and diagnostics
- ✅ Production-ready deployment architecture

**Perfect for:**
- 🎓 Course demonstrations
- 🔬 Research presentations  
- 🚀 Production deployments
- 📊 Automated ML pipelines
- 🧪 Model validation workflows

---

## 🚀 Quick Demo Commands

```bash
# One-time setup
docker volume create ai_system

# Demo ANN model
cd scenarios/aiDockerImage && docker-compose up

# Demo OLS model (in new terminal)
cd scenarios/olsDockerImage && docker-compose up

# Watch automatic predictions appear in terminal! 🎉
```