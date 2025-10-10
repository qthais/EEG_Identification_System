from ast import Dict, For
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import os
import re
import tempfile

from app.services.predict_service import predict_random_segment
from app.services.retrain import retrainModel
from app.Utils.time import add_timestamps
from database.models import EEGRecordCreate, EEGStatus, PredictionCreate, PredictionStatus
from database.db import eeg_collection, prediction_collection 
router= APIRouter()
@router.post('/register_eeg')
async def register_eeg(file: UploadFile):
    try:
        match = re.match(r"^S(\d{3})_48s\.edf$", file.filename)
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
        eeg_create = EEGRecordCreate(filename=filename, subject_id=subject_id, path=str(filepath))
        eeg_doc:dict[str, Any] = add_timestamps(eeg_create.model_dump(exclude_none=True))
        insert_res = await eeg_collection.insert_one(eeg_doc)
        inserted_id = insert_res.inserted_id

        # Import here to avoid circular import
        from app.tasks import retrain_model_task
        retrain_started_at = datetime.now(timezone.utc)
        await eeg_collection.update_one(
            {"_id": inserted_id},
            {
                "$set": {
                    "status": EEGStatus.RETRAINING_STARTED.value,
                    "retrain_started_at": retrain_started_at,
                }
            },
        )
        retrain_model_task.delay(filename)   # 👈 ENABLED: async, non-blocking

        return JSONResponse({
            "message": "EEG registered. Retraining started in background",  # 👈 UPDATED: retraining message
            "filename": filename
        })

    except Exception as e:
        print("❌ Error in /register_eeg:", e)

        await eeg_collection.update_one(
            {"_id": inserted_id},
            {
                "$set":{
                    "status": EEGStatus.RETRAIN_FAILED_TO_START,
                    "notes": str(e)
                }
            }
        )

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
        pred_create = PredictionCreate(
            filename=file.filename,
            predicted_class=result.get("predicted_class", ""),
            confidence=float(result.get("confidence", 0.0)),
            raw_prediction=result.get("raw_prediction"),
            segment_shape=result.get("segment_shape"),
            status=PredictionStatus.COMPLETED,  # optional override
        )
        pred_doc = add_timestamps(pred_create.model_dump(exclude_none=True))
        pred_res = await prediction_collection.insert_one(pred_doc)
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
        fail_doc = {
            "filename": file.filename if file and hasattr(file, "filename") else "unknown",
            "predicted_class": "",
            "confidence": 0.0,
            "raw_prediction": {"error": str(e)},
            "timestamp": datetime.now(timezone.utc),
            "status": PredictionStatus.FAILED.value,
        }
        await prediction_collection.insert_one(fail_doc)
        return JSONResponse({
            "message": "Prediction failed",
            "error": str(e)
        }, status_code=500)