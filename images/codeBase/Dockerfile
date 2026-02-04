FROM python:3.12-slim

# Install ALL required packages for both ANN and OLS
RUN pip install --no-cache-dir \
    pandas==3.0.0 \
    numpy==2.4.1 \
    tensorflow==2.20.0 \
    keras==3.13.1 \
    scikit-learn==1.8.0 \
    statsmodels==0.14.6 \
    scipy

# Create directories
RUN mkdir -p /tmp/activationBase /scripts

# Copy activation data
COPY activation_data.csv /tmp/activationBase/
COPY README.md /tmp/activationBase/

# Copy BOTH prediction scripts
COPY predict_ann.py /scripts/
COPY predict_ols.py /scripts/

# Set working directory
WORKDIR /scripts

# Keep container running
CMD ["tail", "-f", "/dev/null"]