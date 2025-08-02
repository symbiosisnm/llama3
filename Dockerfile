# Use official PyTorch Docker image with CUDA for GPU inference and training
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables to improve Python performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create application directory
WORKDIR /app

# Install system dependencies (optional)
# Ensure git-lfs is available for large model files if needed
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Use business_requirements.txt for our custom pipeline
COPY business_requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the repository code into the container
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Set default environment variables for model and vector store paths
ENV MODEL_PATH=/app/business-llama3 \
    CHROMA_DIR=/app/chromadb \
    COLLECTION_NAME=business_docs

# Specify the command to run the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
