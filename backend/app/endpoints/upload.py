from ast import For
from pathlib import Path
from uuid import uuid4
import uuid
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import os
import mne
import re
import numpy as np
from keras.models import load_model
from app.services.eeg_processing import eeg_to_spectrogram
import tempfile

from app.services.predict_service import predict_random_segment
from app.services.retrain import retrainModel
router= APIRouter()

@router.post('/register_eeg')
async def register_eeg(file: UploadFile):
    try:
        match = re.match(r"S(\d{3})_", file.filename)
        if not match:
            return JSONResponse({
                "message": "Invalid filename format. Expected something like 'S001_48s.edf'"
            }, status_code=400)

        subject_id = int(match.group(1)) - 1  # Convert to 0-based index
        print(f"📛 Extracted subject_id: {subject_id}")

        # Setup save path
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        UPLOAD_DIR = BASE_DIR / "app" / "data" / "uploads"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        filename = f"S{subject_id + 1:03d}R01.edf"
        filepath = UPLOAD_DIR / filename
        print(filepath)
        if filepath.exists():
            return JSONResponse({
                "message": f"Duplicate file already exists: {filename}",
                "filename": filename
            }, status_code=409)

        with open(filepath, "wb") as f:
            f.write(await file.read())

        print(f"✅ Saved EEG file to {filepath}")

        # Call retrain
        print("🔁 Starting retraining...")
        test_accuracy = retrainModel()
        # Return response
        return JSONResponse({
            "message": "EEG registered and model retrained successfully",
            "filename": filename,
            "test_accuracy": test_accuracy
        })

    except Exception as e:
        print("❌ Error in /register_eeg:", e)
        return JSONResponse({
            "message": "Error during registration or retraining",
            "error": str(e)
        }, status_code=500)
@router.post("/login_eeg")
async def login_eeg(file: UploadFile):
    try:
        print(f"LOGIN EEG: received file {file.filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        print(f"📥 Temp file saved: {tmp_path}")

        result = predict_random_segment(tmp_path)

        os.remove(tmp_path)

        return {
            "success": True,
            "message": "Prediction successful",
            "data": {
                "confidence": result["confidence"],
                "predicted_class": result["predicted_class"],
                "raw_prediction": result["raw_prediction"],
                "segment_shape": result["segment_shape"]
            }
        }

    except Exception as e:
        print("❌ Error in /login_eeg:", e)
        return JSONResponse({
            "message": "Prediction failed",
            "error": str(e)
        }, status_code=500)