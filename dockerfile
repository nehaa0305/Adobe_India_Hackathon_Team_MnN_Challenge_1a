# Use a slim Python 3.10 base image for smaller size
FROM python:3.10-slim

# Set environment variables for non-interactive apt-get installs
ENV DEBIAN_FRONTEND=noninteractive

# Update apt-get and install all necessary system dependencies
# This includes tesseract, poppler-utils, git, and libraries required by
# OpenCV (used by layoutparser) and Detectron2 for rendering/display.
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    poppler-utils \
    git \
    build-essential \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 && \
    # Clean up apt cache to keep the image size down
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
# This allows Docker to cache this layer if requirements.txt doesn't change
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# Using --no-cache-dir to prevent caching pip packages locally in the image
# Using --upgrade pip to ensure pip itself is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your Python application code into the container
# Assuming your main script is named `process_pdfs.py`
COPY . .

# Default command to run your application when the container starts
# This will execute your `process_pdfs.py` script
CMD ["python", "process_pdfs.py"]
