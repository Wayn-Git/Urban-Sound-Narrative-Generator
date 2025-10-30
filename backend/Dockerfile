# Lightweight Python base with Debian (writable for apt)
FROM python:3.12-slim

# Install system deps (FFmpeg for pydub/audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create audio dir
RUN mkdir -p audio

# Expose port (Render uses $PORT, but CMD overrides)
EXPOSE $PORT

# Run with uvicorn (use env var for port)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $$PORT"]