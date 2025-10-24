# CIF Processing Integration Guide

This guide explains how the CIF processing script has been integrated into your crystal structure prediction web server.

## Overview

Your CIF processing script has been successfully integrated into a full-stack web application with the following components:

- **Backend**: FastAPI service with CIF processing endpoints
- **Frontend**: React application with file upload and structure visualization
- **Integration**: Seamless communication between frontend and backend

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   CIF Processor │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (Your Script) │
│                 │    │                 │    │                 │
│ • File Upload   │    │ • CIF Processing│    │ • Structure     │
│ • Structure     │    │ • Job Management│    │   Analysis      │
│   Visualization │    │ • Model         │    │ • Feature       │
│ • Results       │    │   Inference    │    │   Extraction    │
│   Display       │    │ • API Endpoints │    │ • Graph         │
└─────────────────┘    └─────────────────┘    │   Construction  │
                                              └─────────────────┘
```

## Key Features Implemented

### 1. CIF File Processing
- **Upload**: Users can drag-and-drop or browse for CIF files
- **Validation**: Client-side and server-side file validation
- **Processing**: Real-time CIF parsing and structure analysis
- **Feedback**: Immediate structure information display

### 2. Structure Information Extraction
- **Formula**: Chemical formula extraction
- **Properties**: Density, volume, space group information
- **Lattice**: Lattice parameters (a, b, c, α, β, γ)
- **Statistics**: Number of atoms, elements, bonds

### 3. Model Integration
- **Data Preparation**: CIF → PyTorch Geometric Data conversion
- **Feature Extraction**: Node and edge features for ML model
- **Inference**: Model prediction pipeline
- **Interpretation**: Results classification and analysis

### 4. User Interface
- **File Upload**: Intuitive drag-and-drop interface
- **Structure Preview**: 3D visualization placeholder
- **Information Display**: Detailed structure properties
- **Job Management**: Submit and track prediction jobs

## File Structure

```
webserver/
├── backend/                    # FastAPI backend service
│   ├── main.py               # Main FastAPI application
│   ├── cif_processor.py      # Your CIF processing script
│   ├── requirements.txt      # Python dependencies
│   ├── start_backend.sh     # Startup script
│   └── README.md            # Backend documentation
├── src/                      # React frontend
│   ├── components/
│   │   ├── FileUpload.tsx   # Enhanced file upload component
│   │   └── StructureViewer.tsx # Structure visualization
│   ├── lib/
│   │   └── api-client.ts    # Updated API client
│   └── pages/
│       └── Predict.tsx       # Updated prediction page
└── README.md                 # This integration guide
```

## Getting Started

### 1. Start the Backend

```bash
cd backend
chmod +x start_backend.sh
./start_backend.sh
```

The backend will start on `http://localhost:8000`

### 2. Start the Frontend

```bash
npm install
npm run dev
```

The frontend will start on `http://localhost:5173`

### 3. Test the Integration

1. Open `http://localhost:5173/predict`
2. Upload a CIF file
3. Observe the structure information extraction
4. Submit for prediction

## API Endpoints

### CIF Processing
- `POST /api/v1/process-cif` - Process CIF file and extract structure info

### Job Management
- `POST /api/v1/submit` - Submit prediction job
- `GET /api/v1/status/{job_id}` - Get job status
- `GET /api/v1/result/{job_id}` - Get job results

## Integration Points

### 1. File Upload Component (`FileUpload.tsx`)
- **Enhanced**: Now processes CIF files with backend
- **Features**: Real-time processing, error handling, structure info display
- **API**: Calls `apiClient.processCIF()` for file processing

### 2. API Client (`api-client.ts`)
- **New Method**: `processCIF()` for CIF file processing
- **Updated Types**: Enhanced interfaces for structure information
- **Error Handling**: Comprehensive error management

### 3. Predict Page (`Predict.tsx`)
- **Integration**: Connects file upload with job submission
- **State Management**: Tracks CIF processing results
- **Workflow**: Complete prediction pipeline

### 4. Structure Viewer (`StructureViewer.tsx`)
- **Enhanced**: Displays extracted structure information
- **Visualization**: Placeholder for 3D structure rendering
- **Details**: Comprehensive structure property display

## CIF Processing Pipeline

1. **Upload**: User uploads CIF file via drag-and-drop
2. **Validation**: Client-side file type validation
3. **Processing**: Backend processes CIF using your script
4. **Extraction**: Structure information extracted
5. **Display**: Information shown in UI components
6. **Submission**: User submits for prediction
7. **Inference**: Model processes structure data
8. **Results**: Predictions returned and displayed

## Error Handling

The integration includes comprehensive error handling:

- **File Validation**: Invalid file types rejected
- **CIF Parsing**: Malformed CIF files handled gracefully
- **Processing Errors**: Structure processing failures caught
- **Network Issues**: API communication errors managed
- **User Feedback**: Clear error messages displayed

## Customization

### Adding Your Model

Replace the mock model in `backend/main.py`:

```python
# Replace MockModel with your actual model
class YourModel:
    def eval(self):
        # Your model evaluation setup
        pass
    
    def __call__(self, data):
        # Your model inference
        return {
            'formation_energy': your_prediction,
            'band_gap': your_prediction
        }
```

### Enhancing Visualization

Add 3D structure visualization by integrating:
- **3Dmol.js**: For molecular visualization
- **NGL Viewer**: For protein/structure visualization
- **Three.js**: For custom 3D rendering

### Extending Features

- **Batch Processing**: Handle multiple CIF files
- **Advanced Analysis**: Additional structure properties
- **Export Options**: Download processed data
- **User Accounts**: Authentication and job history

## Production Considerations

### Security
- File size limits
- File type validation
- Input sanitization
- Rate limiting

### Performance
- Async processing
- Caching
- Database optimization
- Load balancing

### Monitoring
- Logging
- Error tracking
- Performance metrics
- Health checks

## Troubleshooting

### Common Issues

1. **Backend Not Starting**
   - Check Python version (3.8+)
   - Verify dependencies installed
   - Check port availability

2. **CIF Processing Errors**
   - Validate CIF file format
   - Check file size limits
   - Review error logs

3. **Frontend Connection Issues**
   - Verify backend is running
   - Check CORS settings
   - Validate API endpoints

### Debug Mode

Enable debug mode for development:

```bash
# Backend
uvicorn main:app --reload --log-level debug

# Frontend
npm run dev -- --debug
```

## Next Steps

1. **Replace Mock Model**: Integrate your actual trained model
2. **Add 3D Visualization**: Implement structure rendering
3. **Enhance UI**: Improve user experience
4. **Add Testing**: Unit and integration tests
5. **Deploy**: Production deployment setup

## Support

For issues or questions:
1. Check the backend logs
2. Review API documentation at `/docs`
3. Validate CIF file format
4. Test with sample CIF files

The integration is now complete and ready for your crystal structure prediction model!



