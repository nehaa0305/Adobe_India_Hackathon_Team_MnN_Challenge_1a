FROM --platform=linux/amd64 python:3.11

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir notebook nbconvert

# Copy notebook to container
COPY llama.ipynb .

# Convert notebook to script and run it
CMD ["sh", "-c", "jupyter nbconvert --to script llama.ipynb && python llama.py"]
