# Flower Classification using ANN

**Authors:** Devang Thaker & Krish Manvar

**Course:** M. Grum: Advanced AI-based Application Systems  
**Institution:** Junior Chair for Business Information Science, esp. AI-based Application Systems, University of Potsdam

## Project Description
This project implements an Artificial Neural Network (ANN) to classify different flower species based on their features. We scrape flower images/data from the internet, preprocess them, train both an ANN and OLS model, and deploy the solution using Docker containers.

## Dataset

---

## 🐳 Docker Images

### Pull Commands

Pull the Docker images from DockerHub:
```bash
# Pull learning base image
docker pull krish6447/learningbase_flower-classification:latest


# Pull activation base image
docker pull krish6447/activationbase_flower-classification:latest
```

### DockerHub Repositories
- **Learning Base:** https://hub.docker.com/r/krish6447/learningbase_flower-classification
- **Activation Base:** https://hub.docker.com/r/krish6447/activationbase_flower-classification

### Quick Test

Run containers individually:
```bash
# Test learning base
docker run --rm krish6447/learningbase_flower-classification

# Test activation base
docker run --rm krish6447/activationbase_flower-classification
```

### Docker Compose Usage
```bash
# Create external volume (first time only)
docker volume create ai_system

# Start all services
docker-compose up

# In another terminal - access data
docker-compose exec learningbase ls -la /tmp/learningBase/train/
docker-compose exec activationbase ls -la /tmp/activationBase/

# Stop services
docker-compose down
```

### Image Contents

**learningBase Image:**
- `/tmp/learningBase/train/training_data.csv` - Training dataset (80%)
- `/tmp/learningBase/validation/test_data.csv` - Test dataset (20%)
- `/tmp/learningBase/README.md` - Documentation

**activationBase Image:**
- `/tmp/activationBase/activation_data.csv` - Activation dataset (5 samples)
- `/tmp/activationBase/README.md` - Documentation

### Technical Details
- **Base Image:** busybox:latest
- **Volume:** ai_system (external)
- **Network:** flower_network (bridge)
- **License:** AGPL-3.0

---