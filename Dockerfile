# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System performance & logging environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KOKORO_VOICE=af_bella \
    KOKORO_LANG=en-us \
    KOKORO_SPEED=1.0

WORKDIR /app

# Install audio processing runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    espeak-ng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir kokoro-onnx

# Copy source code and documentation
COPY streaming_chunker.py ./
COPY tts_worker.py ./
COPY voice_agent.py ./
COPY voice_service.py ./
COPY demo.py ./
COPY architecture.md ./
COPY README.md ./
COPY docs/ ./docs/

# Create directory for audio chunk outputs
RUN mkdir -p /app/chats && chmod 777 /app/chats

EXPOSE 8001

# Default: Start the FastAPI Voice Microservice
# To run the interactive CLI demo instead: docker run --rm -it <image> python3 demo.py
CMD ["uvicorn", "voice_service:app", "--host", "0.0.0.0", "--port", "8001"]

