FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-hin \
    poppler-utils \
    git \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY process_pdfs.py .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command
CMD ["python", "process_pdfs.py"]
