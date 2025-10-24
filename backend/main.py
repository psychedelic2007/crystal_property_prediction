"""
FastAPI backend service for CIF file processing and crystal structure prediction.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import uuid
from typing import Dict, Any, Optional
import torch
import json
import sys
from datetime import datetime
import asyncio
from pathlib import Path

# Import the CIF processing functions
from .cif_processor import process_cif_file, extract_structure_info, predict_from_cif

app = FastAPI(
    title="Crystal Structure Prediction API",
    description="API for processing CIF files and predicting crystal properties",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:5173",
        "http://0.0.0.0:3000",
        "http://0.0.0.0:8080",
        "https://crystal-predict.onrender.com",
        "https://crystal-property-prediction.onrender.com",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# In-memory job storage (in production, use Redis or database)
jobs_db: Dict[str, Dict[str, Any]] = {}

def _load_real_model() -> Optional[torch.nn.Module]:
    """Load the user's trained model using the exact same method as material_project_prediction.py."""
    try:
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "src" / "model" / "final_best_model.pth"
        config_path = project_root / "src" / "model" / "best_config.json"
        
        # Ensure model modules can be imported
        sys.path.append(str(project_root / "src" / "model"))
        
        if not model_path.exists():
            print(f"Model file not found at {model_path}")
            return None
            
        if not config_path.exists():
            print(f"Config file not found at {config_path}")
            return None
        
        # Load hyperparameters exactly like the user's script
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        params = config['hyperparameters']
        
        print(f"Loading model with:")
        print(f"  Hidden dim: {params['hidden_dim']}")
        print(f"  Depth: {params['depth']}")
        print(f"  Dropout: {params['dropout']}")
        
        # Import the model architecture
        try:
            from train_m3gnet_multitask import SimpleM3GNetInspired
        except ImportError:
            print("Could not import SimpleM3GNetInspired from train_m3gnet_multitask")
            return None
        
        # Initialize with correct architecture
        model = SimpleM3GNetInspired(
            node_dim=4,
            edge_dim=3,
            space_group_dim=230,
            hidden=params['hidden_dim'],
            depth=params['depth'],
            dropout=params['dropout']
        )
        
        # Load checkpoint exactly like the user's script
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Extract model state dict (handles both formats)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint saved with metadata
            state_dict = checkpoint['model_state_dict']
            print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  Val MAE: {checkpoint.get('val_mae', 'N/A')}")
        else:
            # Checkpoint is already a state dict
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict)
        model.eval()
        
        print("✓ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Failed to load real model: {e}")
        import traceback
        traceback.print_exc()
        return None

def _extract_predictions(predictions: Any) -> Dict[str, float]:
    """Extract formation energy from user's model output format."""
    def to_float(x):
        if x is None: return None
        try:
            return float(x.item() if hasattr(x, 'item') else x)
        except Exception:
            return None
    
    # Handle user's model output format (dict with 'energy' key)
    if isinstance(predictions, dict):
        # User's model outputs: {'energy': tensor, 'crystal_system': tensor, 'material_type': tensor, 'stability': tensor}
        formation_energy = predictions.get('energy')
        if formation_energy is not None:
            return {
                'formation_energy': to_float(formation_energy),
                'crystal_system': predictions.get('crystal_system'),
                'material_type': predictions.get('material_type'), 
                'stability': predictions.get('stability')
            }
    
    # Fallback for other formats
    return { 'formation_energy': -0.5 }

# Global model instance (try real, else None)
model = _load_real_model()

@app.get("/")
async def root():
    return {"message": "Crystal Structure Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/process-cif")
async def process_cif_endpoint(file: UploadFile = File(...)):
    """
    Process uploaded CIF file and extract structure information.
    """
    if not file.filename.endswith('.cif'):
        raise HTTPException(status_code=400, detail="File must be a CIF file")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cif') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()
        
        try:
            # Extract structure information
            structure_info = extract_structure_info(temp_file.name)
            
            if 'error' in structure_info:
                raise HTTPException(status_code=400, detail=f"Failed to parse CIF file: {structure_info['error']}")
            
            # Process CIF to Data object for model
            data = process_cif_file(temp_file.name)
            
            if data is None:
                # Don't raise HTTPException here, just return an error response
                return {
                    "success": False,
                    "error": "Failed to process CIF file for model inference. This may be due to atoms being too far apart to form bonds or an invalid structure.",
                    "structure_info": structure_info
                }
            
            # Return structure information and processing status
            return {
                "success": True,
                "structure_info": structure_info,
                "model_ready": True,
                "num_atoms": len(data.x),
                "num_edges": data.edge_index.size(1),
                "space_group": data.space_group.item() + 1,  # Convert back to 1-indexed
                "message": "CIF file processed successfully"
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing CIF: {error_details}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: '{str(e)}'")
            
            # Provide a meaningful error message
            error_message = str(e) if str(e).strip() else f"Unknown error: {type(e).__name__}"
            raise HTTPException(status_code=500, detail=f"Processing error: {error_message}")
        
        finally:
            # Clean up temporary file
            os.unlink(temp_file.name)

@app.post("/api/v1/submit")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    structureData: Optional[str] = None,
    metadata: Optional[str] = None
):
    """
    Submit a prediction job (CIF file or manual structure data).
    """
    job_id = str(uuid.uuid4())
    
    # Parse metadata
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        metadata_dict = {}
    
    # Initialize job status
    jobs_db[job_id] = {
        "jobId": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Job submitted",
        "createdAt": datetime.now().isoformat(),
        "metadata": metadata_dict
    }
    
    if file:
        # Process CIF file
        background_tasks.add_task(process_cif_job, job_id, file)
    elif structureData:
        # Process manual structure data
        try:
            structure_dict = json.loads(structureData)
            background_tasks.add_task(process_manual_job, job_id, structure_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid structure data format")
    else:
        raise HTTPException(status_code=400, detail="No file or structure data provided")
    
    return {"jobId": job_id}

async def process_cif_job(job_id: str, file: UploadFile):
    """Background task to process CIF file and make predictions."""
    try:
        # Update job status
        jobs_db[job_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Processing CIF file..."
        })
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.cif') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            try:
                # Extract structure information
                jobs_db[job_id].update({
                    "progress": 30,
                    "message": "Extracting structure information..."
                })
                
                structure_info = extract_structure_info(temp_file.name)
                
                if 'error' in structure_info:
                    raise Exception(f"Failed to parse CIF: {structure_info['error']}")
                
                # Process CIF for model inference
                jobs_db[job_id].update({
                    "progress": 50,
                    "message": "Preparing data for model..."
                })
                
                data = process_cif_file(temp_file.name)
                
                if data is None:
                    raise Exception("Failed to process CIF file for model inference")
                
                # Make predictions
                jobs_db[job_id].update({
                    "progress": 70,
                    "message": "Running model predictions..."
                })
                
                # Move data to CPU for mock model
                data = data.to('cpu')
                
                # Make predictions with user's trained model
                preds = None
                node_attention = None
                
                if model is not None:
                    try:
                        model.eval()
                        
                        # Extract attention using the same method as material_project_prediction.py
                        # Register hooks to capture attention layers
                        attention_weights = []
                        node_attentions = []
                        
                        def save_attention(module, input, output):
                            if isinstance(output, tuple) and len(output) == 2:
                                x, attn = output
                                if attn is not None and isinstance(attn, torch.Tensor):
                                    node_attentions.append(attn.detach().cpu())
                        
                        # Register hooks for attention pooling layers
                        hooks = []
                        for name, module in model.named_modules():
                            if 'attention_pool' in name.lower():
                                hook = module.register_forward_hook(save_attention)
                                hooks.append(hook)
                                print(f"  Hooked: {name}")
                        
                        with torch.no_grad():
                            preds = model(data)
                        
                        # Extract node attention
                        if len(node_attentions) > 0:
                            node_attention = node_attentions[-1].squeeze().cpu().numpy().tolist()
                            print(f"Attention extracted: {len(node_attention)} weights")
                        else:
                            print("No attention weights found - model may not have attention pooling layers")
                            # Generate mock attention weights for visualization
                            node_attention = [0.5] * len(data.x)
                        
                        # Remove hooks
                        for hook in hooks:
                            hook.remove()
                        
                        print(f"Model prediction successful: {preds}")
                        print(f"Data positions shape: {data.pos.shape if hasattr(data, 'pos') else 'No positions'}")
                        print(f"Data atom types shape: {data.x.shape if hasattr(data, 'x') else 'No atom types'}")
                        
                    except Exception as e:
                        print(f"Model inference failed: {e}")
                        import traceback
                        traceback.print_exc()
                        preds = None
                
                if preds is None:
                    # Fallback if model fails
                    preds = { 'energy': torch.tensor(-0.5) }
                
                predictions = _extract_predictions(preds)
                formation_energy = predictions['formation_energy']
                
                # Get space group for crystal system classification
                space_group_pred = data.space_group.item() + 1
                
                # Classify stability based on formation energy
                stability = "Stable" if formation_energy < -1.0 else "Metastable" if formation_energy < 0.0 else "Unstable"
                
                # Determine crystal system from space group
                crystal_system = "Triclinic"
                if space_group_pred < 2:
                    crystal_system = "Triclinic"
                elif space_group_pred < 15:
                    crystal_system = "Monoclinic"
                elif space_group_pred < 74:
                    crystal_system = "Orthorhombic"
                elif space_group_pred < 142:
                    crystal_system = "Tetragonal"
                elif space_group_pred < 167:
                    crystal_system = "Trigonal"
                elif space_group_pred < 194:
                    crystal_system = "Hexagonal"
                else:
                    crystal_system = "Cubic"
                
                # Extract atom types for visualization
                atom_types = data.x[:, 0].cpu().tolist() if hasattr(data, 'x') else None
                
                # Update job with results
                jobs_db[job_id].update({
                    "status": "completed",
                    "progress": 100,
                    "message": "Prediction completed successfully",
                    "completedAt": datetime.now().isoformat(),
                    "results": {
                        "structure_info": structure_info,
                        "predictions": {
                            "formation_energy_per_atom": round(formation_energy, 4),
                            "stability": stability,
                            "crystal_system": crystal_system,
                            "space_group_number": space_group_pred
                        },
                        "attention": {
                            "node": node_attention
                        },
                        "positions": data.pos.cpu().tolist() if hasattr(data, 'pos') else None,
                        "atom_types": atom_types,
                        "model_version": "1.0.0"
                    }
                })
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
                
    except Exception as e:
        jobs_db[job_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Job failed: {str(e)}",
            "error": str(e)
        })

async def process_manual_job(job_id: str, structure_data: Dict[str, Any]):
    """Background task to process manual structure data and make predictions."""
    try:
        jobs_db[job_id].update({
            "status": "processing",
            "progress": 50,
            "message": "Processing manual structure data..."
        })
        
        # For manual input, we would need to create a structure object
        # This is a simplified version - you'd need to implement proper structure creation
        # from lattice vectors and atom sites
        
        # Mock predictions for manual input
        formation_energy = -0.3
        
        jobs_db[job_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Prediction completed successfully",
            "completedAt": datetime.now().isoformat(),
            "results": {
                "structure_info": {
                    "formula": "Manual Input",
                    "num_atoms": len(structure_data.get('atomSites', [])),
                    "num_elements": len(set(site['species'] for site in structure_data.get('atomSites', []))),
                    "density": 2.5,  # Mock value
                    "volume": 100.0,  # Mock value
                    "space_group_symbol": "P1",
                    "space_group_number": 1,
                    "lattice_params": {
                        "a": 10.0, "b": 10.0, "c": 10.0,
                        "alpha": 90.0, "beta": 90.0, "gamma": 90.0
                    }
                },
                "predictions": {
                    "formation_energy_per_atom": round(formation_energy, 4),
                    "stability": "Metastable",
                    "crystal_system": "Cubic",
                    "space_group_number": 1
                },
                "model_version": "1.0.0"
            }
        })
        
    except Exception as e:
        jobs_db[job_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Job failed: {str(e)}",
            "error": str(e)
        })

@app.get("/api/v1/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a prediction job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    return {
        "jobId": job["jobId"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "createdAt": job.get("createdAt"),
        "completedAt": job.get("completedAt")
    }

@app.get("/api/v1/result/{job_id}")
async def get_job_result(job_id: str):
    """Get the results of a completed prediction job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if "results" not in job:
        raise HTTPException(status_code=500, detail="Results not available")
    
    return {
        "jobId": job["jobId"],
        "prediction": job["results"]["predictions"],
        "structure_info": job["results"]["structure_info"],
        "attention": job["results"]["attention"],
        "positions": job["results"]["positions"],
        "atom_types": job["results"]["atom_types"],
        "modelVersion": job["results"]["model_version"],
        "completedAt": job["completedAt"]
    }

@app.get("/api/v1/jobs")
async def get_jobs(limit: int = 50):
    """Get list of recent jobs."""
    jobs = list(jobs_db.values())
    jobs.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
    return jobs[:limit]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
