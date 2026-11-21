# Face Audit System — Goggles Wrapper API
# Serves POST /validate on port 8000
# Models are mounted at runtime via Docker volume (not baked in)

FROM python:3.11-slim

WORKDIR /app

# system libs needed by opencv, onnxruntime, insightface
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# install Python deps first (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code (models mounted separately via volume)
COPY main.py .
COPY goggles_wrapper_api.py .
COPY fake_albumentations.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "goggles_wrapper_api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
