# SignalScope — Cloud Deployment (Hugging Face Spaces)
# =====================================================
# Uses CPU inference (no DirectML on cloud).
# SIGNALSCOPE_DEVICE=cpu is set via HF Spaces secrets.

FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_cloud.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_cloud.txt

# Copy application code
COPY app/         ./app/
COPY src/         ./src/
COPY configs/     ./configs/
COPY frontend/    ./frontend/
COPY outputs/     ./outputs/
COPY models/      ./models/
COPY data/signalscope.db ./data/signalscope.db 2>/dev/null || true

# Create data directory (SQLite will create db here if missing)
RUN mkdir -p data logs

# HF Spaces uses port 7860
EXPOSE 7860

# CPU inference — DirectML not available on cloud
ENV SIGNALSCOPE_DEVICE=cpu

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
