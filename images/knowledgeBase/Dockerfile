FROM busybox:latest

# Create directory structure
RUN mkdir -p /tmp/knowledgeBase

# Copy AI models
COPY currentAiSolution.h5 /tmp/knowledgeBase/
COPY currentOlsSolution.pkl /tmp/knowledgeBase/
COPY currentOlsSolution.xml /tmp/knowledgeBase/
COPY README.md /tmp/knowledgeBase/

# Keep container running
CMD ["tail", "-f", "/dev/null"]