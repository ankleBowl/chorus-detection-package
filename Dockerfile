# Use a more recent Python base image
FROM python:3.11-slim

# Set environment variables to improve Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install FFmpeg and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies with optimizations for binary packages
RUN pip install --upgrade pip && \
    pip install --no-binary=:all: --only-binary=numpy,scipy,matplotlib wheel && \
    pip install -r requirements.txt

# Copy the source code
COPY src/ .

# Copy the model file to the working directory
COPY models/CRNN/best_model_V3.h5 /app/models/CRNN/

# Create directories for input and output
RUN mkdir -p /app/input /app/output

# Set volume mount points
VOLUME ["/app/input", "/app/output"]

# Run the script when the container starts
CMD ["python", "chorus_finder.py"]