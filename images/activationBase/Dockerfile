FROM busybox:latest

# Create required directory
RUN mkdir -p /tmp/activationBase

# Copy activation data file
COPY /data/activation_data.csv /tmp/activationBase/

# Copy README documentation
COPY README.md /tmp/activationBase/

# Set appropriate permissions
RUN chmod -R 755 /tmp/activationBase

# Set working directory
WORKDIR /tmp/activationBase

# Default command to list contents
CMD ["sh", "-c", "echo 'Activation Base Image Ready' && ls -la /tmp/activationBase/ && cat /tmp/activationBase/activation_data.csv"]