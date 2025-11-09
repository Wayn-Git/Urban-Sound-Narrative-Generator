# Base image: lightweight but fully compatible with Whisper
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Torch CPU version (Railway free tier usually doesn’t support GPU)
RUN pip install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of your app code
COPY . .

# Create audio directory (safe for uploads/temp files)
RUN mkdir -p /app/audio

# Railway sets PORT env automatically — use it dynamically
EXPOSE 7860
ENV PORT=7860

# Run FastAPI server
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
