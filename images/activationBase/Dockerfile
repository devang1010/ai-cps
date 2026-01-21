FROM busybox:latest

# Metadata
LABEL maintainer="Krish-krishmanvar123@gmail.com"
LABEL description="Activation Base Image for Flower Classification AI Project"
LABEL course="Advanced AI-based Application Systems"
LABEL university="University of Potsdam"

# Create directory
RUN mkdir -p /tmp/activationBase

# Copy data file
COPY data/activation_data.csv /tmp/activationBase/

# Copy README
COPY README.md /tmp/activationBase/

# Set working directory
WORKDIR /tmp/activationBase

# Default command - list contents
CMD ["ls", "-la", "/tmp/activationBase/"]