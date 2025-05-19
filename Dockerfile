# Use a Python base image consistent with environment.yml
FROM python:3.8-slim

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
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the source code and CLI script
COPY src/ ./src/
COPY chorus-detection-CLI.py .
COPY streamlit_app.py .

# Copy the model file to the working directory
RUN mkdir -p /app/models/CRNN
COPY models/CRNN/best_model_V3.h5 /app/models/CRNN/

# Create directories for input and output
RUN mkdir -p /app/input /app/output

# Set volume mount points
VOLUME ["/app/input", "/app/output"]

# Expose port for Streamlit (if used)
EXPOSE 8501

# Run the CLI script by default when the container starts
CMD ["python", "chorus-detection-CLI.py"]