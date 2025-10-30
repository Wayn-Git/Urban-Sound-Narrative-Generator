# Use official Python runtime as base image
FROM python:3.12-slim

# Install system dependencies (FFmpeg for pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create audio dir
RUN mkdir -p audio

# Expose port
EXPOSE 10000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]