FROM python:3.10.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render uses $PORT environment variable)
EXPOSE 10000

# Start FastAPI - Use $PORT from Render
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}