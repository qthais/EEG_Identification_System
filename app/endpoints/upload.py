from ast import For
from pathlib import Path
from uuid import uuid4
import uuid
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import os
import mne
import numpy as np
from keras.models import load_model
from app.services.eeg_processing import eeg_to_spectrogram
import tempfile

from app.services.predict_service import predict_random_segment
from app.services.retrain import retrainModel
router= APIRouter()

@router.post('/register_eeg')
async def register_eeg(file: UploadFile, subject_id: int = Form(...)):
    try:
        unique_id = uuid.uuid4().hex[:6]
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        UPLOAD_DIR = BASE_DIR / "app" / "data" / "uploads"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        filename = f"S{subject_id + 1:03d}R01_{unique_id}.edf"
        filepath = UPLOAD_DIR / filename
        print(filepath)

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
            "message": "Prediction successful",
            "confidence": result["confidence"],
            "predicted_class": result["predicted_class"],
            "raw_prediction": result["raw_prediction"],
            "segment_shape": result["segment_shape"]
        }

    except Exception as e:
        print("❌ Error in /login_eeg:", e)
        return JSONResponse({
            "message": "Prediction failed",
            "error": str(e)
        }, status_code=500)