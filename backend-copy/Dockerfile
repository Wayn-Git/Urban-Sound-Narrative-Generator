# Python 3.11 slim (Whisper-compatible)
FROM python:3.11-slim

# Install FFmpeg (required by Whisper)
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Torch CPU wheels (needed for Whisper)
RUN pip install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Copy your app code
COPY . .

# Create audio dir
RUN mkdir -p audio

# Expose port 7860 (required by HF Spaces)
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
