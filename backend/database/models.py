from time import timezone
from typing import Optional, List, Any, Dict
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, EmailStr


# --- Status enums ----------------------------------------------------------
class EEGStatus(str, Enum):
    PENDING_RETRAIN = "pending_retrain"
    RETRAINING_STARTED = "retraining_started"
    RETRAIN_FAILED_TO_START = "retrain_failed_to_start"
    RETRAIN_COMPLETED = "retrain_completed"
    RETRAIN_FAILED = "retrain_failed"


class PredictionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Base model ------------------------------------------------------------
class BaseMongoModel(BaseModel):
    # Use alias "_id" so the model matches Mongo documents;
    # when saving new docs don't include this field (Mongo will create it).
    id: Optional[str] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        # make datetime and Enum json serializable in responses
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value if isinstance(v, Enum) else v,
        }


# --- User models (no is_active, minimal fields) ----------------------------
class User(BaseMongoModel):
    user_id: str = Field(..., description="User unique code (from EDF header, e.g. patientcode)", unique=True)
    user_name: str = Field(..., description="User display name (from EDF header, e.g. patientname)")


class UserCreate(BaseModel):
    user_id: str = Field(..., description="User unique code (from EDF header, e.g. patientcode)", unique=True)
    user_name: str = Field(..., description="User display name (from EDF header, e.g. patientname)")


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(default=None, max_length=100)
    eeg_files: Optional[List[str]] = None


class UserResponse(BaseMongoModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    eeg_files: List[str] = Field(default_factory=list)


# --- EEG record models (for uploads / retraining tracking) -----------------
class EEGRecordBase(BaseMongoModel):
    filename: str
    subject_id: int
    user_id: Optional[str] = Field(default=None, description="User unique code (from EDF header)")
    path: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    status: EEGStatus = Field(default=EEGStatus.PENDING_RETRAIN)
    retrain_started_at: Optional[datetime] = None
    retrain_finished_at: Optional[datetime] = None
    notes: Optional[str] = None


class EEGRecordCreate(BaseModel):
    filename: str
    subject_id: int
    user_id: Optional[str] = Field(default=None, description="User unique code (from EDF header)")
    path: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[EEGStatus] = None  # optional override on create


class EEGRecordResponse(EEGRecordBase):
    pass


# --- Prediction models (inference logs) -----------------------------------
class PredictionBase(BaseMongoModel):
    filename: str
    predicted_class: str
    confidence: float
    raw_prediction: Any
    raw_data: Optional[Any] = Field(default=None, description="EEG raw data used for prediction")
    user_id: Optional[str] = Field(default=None, description="User unique code (from EDF header)")
    segment_shape: Optional[List[int]] = None
    status: PredictionStatus = Field(default=PredictionStatus.COMPLETED)


class PredictionCreate(BaseModel):
    user_id: Optional[str] = Field(default=None, description="User unique code (from EDF header)")
    filename: str
    predicted_class: int
    confidence: float
    raw_prediction: Any
    raw_data: Optional[Any] = Field(default=None, description="EEG raw data used for prediction")
    segment_shape: Optional[List[int]] = None
    status: Optional[PredictionStatus] = None  # optional override on create


class PredictionResponse(PredictionBase):
    pass


# --- Helper: convert Motor/Mongo doc to Pydantic model ---------------------
def mongo_doc_to_model(model_class, doc: Dict):
    """
    Convert a raw Mongo doc (with _id as ObjectId) into a Pydantic model instance
    with string id. Usage:
      model = mongo_doc_to_model(EEGRecordResponse, mongo_doc)
    """
    doc_copy = dict(doc)
    if "_id" in doc_copy:
        doc_copy["_id"] = str(doc_copy["_id"])
    # If status values are stored as strings in DB, Pydantic will coerce them into Enum fields automatically.
    return model_class.parse_obj(doc_copy)
