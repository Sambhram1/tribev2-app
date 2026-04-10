FROM python:3.11-slim

# System deps: ffmpeg for video decoding, libsndfile1 for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install uv (provides uvx, required by tribev2 for audio transcription)
RUN pip install --no-cache-dir uv

# Install tribev2 from source
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/tribev2.git

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
