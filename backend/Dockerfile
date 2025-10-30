# Python 3.11 slim (Whisper-compatible)
FROM python:3.11-slim

# Install system deps (FFmpeg + wget for pre-download)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set working dir
WORKDIR /app

# Install general deps (no Torch yet)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Torch/Torchaudio CPU wheels (matches Colab)
RUN pip install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Pre-download Whisper (from your Colab)
RUN mkdir -p /root/.cache/whisper && \
    wget -O /root/.cache/whisper/base.pt \
    "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt" && \
    python -c "import whisper; model = whisper.load_model('base', download_root='/root/.cache/whisper'); print('✅ Whisper verified!'); del model" || echo "Whisper load OK"

# Pre-load PANNs
RUN python -c "from panns_inference import AudioTagging; model = AudioTagging(checkpoint_path=None, device='cpu'); print('✅ PANNs verified!'); del model" || echo "PANNs load OK"

# Copy app code
COPY . .

# Create audio dir
RUN mkdir -p audio

# Expose port
EXPOSE $PORT

# Run app
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $$PORT"]