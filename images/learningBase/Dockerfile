FROM busybox:latest

# Create required directories
RUN mkdir -p /tmp/learningBase/train
RUN mkdir -p /tmp/learningBase/validation

# Copy data files to the appropriate locations
COPY /data/training_data.csv /tmp/learningBase/train/
COPY /data/test_data.csv /tmp/learningBase/validation/

# Copy README documentation
COPY README.md /tmp/learningBase/

# Set appropriate permissions
RUN chmod -R 755 /tmp/learningBase

# Set working directory
WORKDIR /tmp/learningBase

# Default command to list contents
CMD ["sh", "-c", "echo 'Learning Base Image Ready' && ls -la /tmp/learningBase/train/ && ls -la /tmp/learningBase/validation/"]