# ML Model Inference & Deployment Guide

This document outlines how to deploy the Electronic Taste subgenre classification model for inference, both for local development testing and production containerized hosting.

---

## 0. Downloading Pre-trained Model Weights

Because model weights are ignored by version control, they are hosted publicly on Google Cloud Storage. Collaborators can download the required weights directly before running the local API or scripts:

### Download Links
* **Best Classifier Head (`classifier_best.pt`):** [Download Link](https://storage.googleapis.com/electronic-taste-models-jbroadbent/models/classifier_best.pt)
* **Label Encoder Map (`label_encoder.json`):** [Download Link](https://storage.googleapis.com/electronic-taste-models-jbroadbent/models/label_encoder.json)

### Shell Setup (Automatic Download)
You can download and place the model files into the correct folder structure using the following commands:

```bash
# Create target directory
mkdir -p models

# Download model files
curl -o models/classifier_best.pt https://storage.googleapis.com/electronic-taste-models-jbroadbent/models/classifier_best.pt
curl -o models/label_encoder.json https://storage.googleapis.com/electronic-taste-models-jbroadbent/models/label_encoder.json
```

---

## 1. Local Development Execution

The core inference pipeline is implemented as a **FastAPI** application under `src/api/` and a CLI batch processing tool under `src/scripts/`.

### Run the FastAPI Server
To launch the FastAPI development server with hot-reloading:

```bash
# From the project root with your virtual environment active
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Health Check Endpoint
To verify the model is loaded successfully into memory:

```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "model": "MERT-v1-95M",
  "device": "mps"
}
```

---

## 2. API Prediction Contract

The endpoint accepts raw audio uploads, resamples the audio to 24 kHz mono, processes it through MERT, runs it through the linear classifier head, and outputs the top 3 subgenres.

### POST /predict
* **Content-Type:** `multipart/form-data`
* **Parameters:** `file` (Binary audio file: WAV, MP3, etc.)

#### Querying with cURL:
```bash
curl -X POST -F "file=@/Users/jbroadbent/Code/ElectronicTaste/dummy_track.wav" \
    http://localhost:8000/predict
```

#### Response Payload:
```json
[
  {
    "label": "techno",
    "confidence": 0.6452
  },
  {
    "label": "minimal",
    "confidence": 0.1831
  },
  {
    "label": "deep house",
    "confidence": 0.0927
  }
]
```

---

## 3. CLI Batch Inference Script

To run classification on a directory of files without launching the server, use `ingest_batch.py`:

```bash
python3 src/scripts/ingest_batch.py \
    --audio-dir data/my_tracks \
    --model models/classifier_best.pt \
    --output results.csv
```
This writes a CSV output with top-3 predictions and confidences for each file.

---

## 4. Production Deployment Strategy (Docker)

To deploy to a cloud instance (e.g., GCP Cloud Run, Compute Engine, AWS ECS, or RunPod), wrap the application in a Docker container.

### Production Dockerfile Template

Create a `Dockerfile` in the root of the project:

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for audio reading
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run with Gunicorn using Uvicorn workers for high concurrency
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

### Resource Requirements
* **RAM:** Minimum **4 GB** (the MERT model occupies ~380 MB in memory, but audio tensors and multiprocessing overhead require safety margin).
* **GPU (Optional but recommended for scale):** NVIDIA T4 or L4. Set `--device cuda` when launching or configure the FastAPI singleton model instantiation to detect CUDA.
* **CPU-only:** If running on CPU (e.g. standard Cloud Run), ensure the server is configured with at least 2 vCPUs and the model device is explicitly set to `cpu`.
