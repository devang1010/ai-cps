FROM busybox:latest

# Create directory structure
RUN mkdir -p /tmp/activationBase

# Copy activation data
COPY activation_data.csv /tmp/activationBase/
COPY README.md /tmp/activationBase/

# Keep container running
CMD ["tail", "-f", "/dev/null"]