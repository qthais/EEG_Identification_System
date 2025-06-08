from ast import For
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import os
import mne
import numpy as np
import joblib
from pydantic import FilePath
import tensorflow as tf
from keras.models import load_model
from app.services.eeg_processing import eeg_to_spectrogram
import tempfile
router= APIRouter()

@router.post('/register_eeg')
async def register_eeg(file:UploadFile, subject_id:int=Form(...)):
    print(file)
    return JSONResponse({
        "message": "EEG registered successfully",
        "subject_id": subject_id
    })
@router.post("/login_eeg")
async def login_eeg(file: UploadFile):
    print(f"LOGIN EEG: received file {file.filename}")
    # For now, return fake prediction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name
    print(tmp_path)
    # Now read with MNE
    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    os.remove(tmp_path)
    print(raw.info)

    return {"message": "EEG file processed successfully"}