# Stage 1: Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (build-essential for fast C++ extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source and configuration
COPY src/ ./src/
COPY models/ ./models/
COPY README.md .

# Create output and student_resource directories
RUN mkdir -p output student_resource/dataset

# Default command runs end-to-end inference and validation
CMD ["python", "-m", "src.generate_outputs", "--test-s1", "student_resource/dataset/test/test_source1.tsv", "--test-s2", "student_resource/dataset/test/test_source2.tsv", "--test-s3", "student_resource/dataset/test/test_source3.tsv", "--output-dir", "output"]
