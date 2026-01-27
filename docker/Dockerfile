FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    curl \
    cargo \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files for installation
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install ".[api]" && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY api/ api/
COPY tagger.py .
COPY phonemizer.py .
COPY utils.py .

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY api/ /app/api/
COPY tagger.py /app/
COPY phonemizer.py /app/
COPY utils.py /app/

# Environment variables
ENV PYTHONPATH=/app
ENV REDIS_HOST=redis
ENV MODEL_PATH=models/luo-tts/female/checkpoint_180000.pth
ENV CONFIG_PATH=models/luo-tts/female/config.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
