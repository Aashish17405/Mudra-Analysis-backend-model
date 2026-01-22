import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline components
# Ensure current directory is in sys.path if needed
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import run_inference

app = FastAPI(
    title="Mudra Analysis API",
    description="API for analyzing Bharatanatyam dance videos for Mudras and Steps.",
    version="1.0.0"
)

# Configuration
UPLOAD_DIR = Path("data/temp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/status")
async def get_status():
    """Check the health of the API."""
    return {"status": "healthy", "message": "Mudra Analysis API is running."}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...), use_mudra: bool = True):
    """
    Upload a video file to analyze mudras and dance steps.
    Returns the JSON inference result.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    # Generate unique ID for this task
    task_id = str(uuid.uuid4())
    temp_video_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    output_json_path = UPLOAD_DIR / f"{task_id}_result.json"

    try:
        # Save uploaded file
        with temp_video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing video: {file.filename} (Task ID: {task_id})")

        # Run inference
        # Note: run_inference saves to output_json_path
        run_inference(str(temp_video_path), str(output_json_path), use_mudra_model=use_mudra)

        # Read results
        if not output_json_path.exists():
            raise HTTPException(status_code=500, detail="Inference failed to generate results.")

        with output_json_path.open("r") as f:
            result_data = json.load(f)

        return result_data

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup temp files
        if temp_video_path.exists():
            temp_video_path.unlink()
        if output_json_path.exists():
            output_json_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
