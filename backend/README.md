# Crystal Structure Prediction Backend

This backend service integrates your CIF processing script with a FastAPI web server to handle crystal structure prediction requests.

## Features

- **CIF File Processing**: Upload and process CIF files using your existing script
- **Structure Information Extraction**: Extract crystal structure properties (formula, space group, lattice parameters, etc.)
- **Model Inference**: Process structures for machine learning model predictions
- **Job Management**: Asynchronous job processing with status tracking
- **RESTful API**: Clean API endpoints for frontend integration

## Setup

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Server

#### Option 1: Using the startup script
```bash
chmod +x start_backend.sh
./start_backend.sh
```

#### Option 2: Manual startup
```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /` - Basic API information
- `GET /health` - Health check endpoint

### CIF Processing
- `POST /api/v1/process-cif` - Process uploaded CIF file and extract structure information

### Job Management
- `POST /api/v1/submit` - Submit a prediction job (CIF file or manual structure data)
- `GET /api/v1/status/{job_id}` - Get job status
- `GET /api/v1/result/{job_id}` - Get job results
- `GET /api/v1/jobs` - Get list of recent jobs

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Integration with Frontend

The frontend is configured to connect to the backend at `http://localhost:8000`. Update the `VITE_API_BASE_URL` environment variable if running on a different port.

## File Structure

```
backend/
├── main.py                 # FastAPI application
├── cif_processor.py        # CIF processing functions (your script)
├── requirements.txt        # Python dependencies
├── start_backend.sh       # Startup script
└── README.md              # This file
```

## CIF Processing Pipeline

1. **File Upload**: CIF file is uploaded via multipart form data
2. **Structure Parsing**: Uses pymatgen to parse CIF and extract structure
3. **Graph Construction**: Creates crystal structure graph using CrystalNN
4. **Feature Extraction**: Extracts node and edge features for ML model
5. **Data Preparation**: Converts to PyTorch Geometric Data format
6. **Model Inference**: Runs predictions (currently mock model)
7. **Result Interpretation**: Classifies stability, material type, crystal system

## Error Handling

The API includes comprehensive error handling for:
- Invalid file formats
- CIF parsing errors
- Structure processing failures
- Model inference errors

## Development Notes

- The current implementation uses a mock model for demonstration
- Replace the `MockModel` class in `main.py` with your actual trained model
- Job storage is currently in-memory (use Redis or database for production)
- CORS is configured for local development (update for production)

## Production Deployment

For production deployment:

1. Use a production ASGI server like Gunicorn with Uvicorn workers
2. Implement proper database storage for job management
3. Add authentication and authorization
4. Configure proper CORS settings
5. Add logging and monitoring
6. Use environment variables for configuration

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed correctly
2. **Port Conflicts**: Change the port in the startup command if 8000 is occupied
3. **CIF Parsing Errors**: Check that CIF files are valid and properly formatted
4. **Memory Issues**: Large structures may require more memory allocation

### Logs

The server logs will show:
- Request/response information
- CIF processing status
- Error messages and stack traces
- Job processing progress



