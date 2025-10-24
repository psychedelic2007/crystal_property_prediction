# CrystalPredict Backend Integration Guide

This frontend is ready to connect to your FastAPI backend. Here's how to integrate everything:

## 🎯 Overview

The React frontend is fully built and ready. You'll deploy the FastAPI backend separately and connect them via API calls.

## 📁 Backend Structure (Deploy Separately)

Create your FastAPI backend with this structure:

```
backend/
├── app/
│   ├── main.py              # FastAPI app with endpoints
│   ├── models/              # ML model loading & inference
│   ├── parsers/             # CIF parsing with pymatgen
│   ├── features/            # Feature extraction
│   ├── tasks/               # Celery task definitions
│   └── storage/             # S3/local file storage
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/                     # Kubernetes manifests
├── requirements.txt
└── .env.example
```

## 🔧 Required Backend Endpoints

Your FastAPI backend should implement these endpoints (already integrated in frontend):

### 1. Submit Job
```python
@app.post("/api/v1/submit")
async def submit_job(
    file: Optional[UploadFile] = File(None),
    structureData: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
) -> dict:
    """
    Accept CIF file or manual structure input
    Returns: {"jobId": "uuid"}
    """
```

### 2. Job Status
```python
@app.get("/api/v1/status/{job_id}")
async def get_status(job_id: str) -> dict:
    """
    Returns: {
        "jobId": str,
        "status": "pending" | "processing" | "completed" | "failed",
        "progress": int,
        "message": str
    }
    """
```

### 3. Get Results
```python
@app.get("/api/v1/result/{job_id}")
async def get_result(job_id: str) -> dict:
    """
    Returns: {
        "jobId": str,
        "prediction": float,
        "uncertainty": float,
        "property": str,
        "featureImportance": list,
        "parityData": dict,
        "modelVersion": str,
        "completedAt": str
    }
    """
```

### 4. Download Report
```python
@app.get("/api/v1/result/{job_id}/report")
async def download_report(job_id: str):
    """
    Returns PDF report as file download
    """
```

### 5. List Jobs
```python
@app.get("/api/v1/jobs")
async def list_jobs(limit: int = 50) -> list:
    """
    Returns list of job status objects
    """
```

## 🚀 Backend Implementation Guide

### Step 1: FastAPI Setup
```python
# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(title="CrystalPredict API", version="1.0.0")

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Step 2: CIF Parsing with Pymatgen
```python
# app/parsers/cif_parser.py
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

def parse_cif_file(file_path: str) -> Structure:
    """Parse CIF file and return pymatgen Structure"""
    parser = CifParser(file_path)
    structure = parser.get_structures()[0]
    return structure

def validate_structure(structure: Structure) -> bool:
    """Validate structure completeness and correctness"""
    # Add your validation logic
    return True
```

### Step 3: Model Inference
```python
# app/models/predictor.py
import torch
import onnxruntime as ort

class CrystalPropertyPredictor:
    def __init__(self, model_path: str, use_gpu: bool = True):
        # Load TorchScript or ONNX model
        if model_path.endswith('.pt'):
            self.model = torch.jit.load(model_path)
        else:
            providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(model_path, providers=providers)
    
    def predict(self, features: dict) -> dict:
        """
        Run inference and return prediction + uncertainty
        """
        # Your model inference logic
        prediction = 5.43  # Example
        uncertainty = 0.12
        
        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "featureImportance": self.get_feature_importance(features)
        }
```

### Step 4: Task Queue (Celery + Redis)
```python
# app/tasks/prediction_tasks.py
from celery import Celery
import redis

celery_app = Celery('crystal_predict', broker='redis://localhost:6379/0')

@celery_app.task(bind=True)
def run_prediction(self, job_id: str, structure_data: dict):
    """
    Background task for prediction
    Updates job status in Redis
    """
    redis_client = redis.Redis()
    
    try:
        # Update status to processing
        redis_client.hset(f"job:{job_id}", "status", "processing")
        
        # Parse structure
        # Extract features
        # Run model inference
        # Save results
        
        # Update status to completed
        redis_client.hset(f"job:{job_id}", "status", "completed")
    except Exception as e:
        redis_client.hset(f"job:{job_id}", "status", "failed")
        redis_client.hset(f"job:{job_id}", "error", str(e))
```

### Step 5: File Storage
```python
# app/storage/s3_storage.py
import boto3
from botocore.config import Config

class FileStorage:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version='s3v4')
        )
        self.bucket = "crystal-predict-files"
    
    def upload_file(self, file_path: str, key: str):
        self.s3_client.upload_file(file_path, self.bucket, key)
    
    def download_file(self, key: str, file_path: str):
        self.s3_client.download_file(self.bucket, key, file_path)
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - MODEL_PATH=/models/model.onnx
    volumes:
      - ./models:/models
    depends_on:
      - redis

  worker:
    build: .
    command: celery -A app.tasks.prediction_tasks worker --loglevel=info
    depends_on:
      - redis
    volumes:
      - ./models:/models

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## ☸️ Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crystal-predict-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crystal-predict
  template:
    metadata:
      labels:
        app: crystal-predict
    spec:
      containers:
      - name: api
        image: your-registry/crystal-predict:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        resources:
          limits:
            nvidia.com/gpu: 1  # Optional GPU support
```

## 🔗 Connecting Frontend to Backend

### Environment Variables
Add to your frontend `.env` file:
```env
VITE_API_BASE_URL=http://localhost:8000
# For production:
# VITE_API_BASE_URL=https://api.crystalpredict.com
```

### Testing the Connection
The frontend already uses the API client in `src/lib/api-client.ts`. Once your backend is running, the frontend will automatically connect.

## 📊 Data Flow

1. **User uploads CIF** → Frontend validates → Sends to `/api/v1/submit`
2. **Backend receives** → Parses with pymatgen → Queues job in Celery
3. **Worker processes** → Extracts features → Runs ML model → Saves results
4. **Frontend polls** → `/api/v1/status/{job_id}` until completed
5. **User views results** → Frontend fetches from `/api/v1/result/{job_id}`

## 🔒 Security Best Practices

1. **Rate Limiting**: Use `slowapi` or similar
2. **File Size Limits**: Max 5MB for CIF files
3. **Input Validation**: Validate all CIF files with pymatgen
4. **CORS**: Only allow your frontend domain
5. **Authentication**: Add JWT tokens for production
6. **Secrets**: Use environment variables, never hardcode

## 📦 Dependencies (requirements.txt)

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
pymatgen==2024.1.26
torch==2.1.2
onnxruntime-gpu==1.16.3  # or onnxruntime for CPU
celery==5.3.4
redis==5.0.1
boto3==1.34.24  # For S3
pydantic==2.5.3
slowapi==0.1.9  # Rate limiting
python-jose[cryptography]==3.3.0  # JWT
```

## 🚦 Quick Start

1. **Clone your backend repo**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Start Redis**: `docker run -d -p 6379:6379 redis:7-alpine`
4. **Run FastAPI**: `uvicorn app.main:app --reload`
5. **Run Celery worker**: `celery -A app.tasks.prediction_tasks worker`
6. **Frontend connects automatically** at `http://localhost:8000`

## 📝 Notes

- The frontend is already fully integrated and will work once your backend is running
- All API endpoints are defined in `src/lib/api-client.ts`
- The frontend handles file uploads, job polling, and result visualization
- For production, deploy backend on AWS/GCP/Azure and update `VITE_API_BASE_URL`

## 🎨 Frontend Features Already Implemented

✅ CIF file drag-and-drop upload with validation
✅ Manual structure input (lattice vectors + atomic coordinates)
✅ 3D structure viewer placeholder (integrate 3Dmol.js)
✅ Job status tracking with real-time updates
✅ Interactive results with Plotly visualizations
✅ PDF report download button
✅ Responsive design with accessibility
✅ Framer Motion animations

## 🔮 Next Steps

1. Deploy your FastAPI backend
2. Configure CORS with your frontend URL
3. Update `VITE_API_BASE_URL` in frontend
4. Test end-to-end workflow
5. Add authentication if needed
6. Scale with Kubernetes

The frontend is production-ready and waiting for your backend! 🚀
