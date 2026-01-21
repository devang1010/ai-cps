FROM busybox:latest

# Metadata
LABEL maintainer="Krish-krishmanvar123@gmail.com"
LABEL description="Learning Base Image for Flower Classification AI Project"
LABEL course="Advanced AI-based Application Systems"
LABEL university="University of Potsdam"

# Create directories
RUN mkdir -p /tmp/learningBase/train
RUN mkdir -p /tmp/learningBase/validation

# Copy data files
COPY data/train/training_data.csv /tmp/learningBase/train/
COPY data/validation/test_data.csv /tmp/learningBase/validation/

# Copy README
COPY README.md /tmp/learningBase/

# Set working directory
WORKDIR /tmp/learningBase

# Default command - list contents
CMD ["ls", "-la", "/tmp/learningBase/"]