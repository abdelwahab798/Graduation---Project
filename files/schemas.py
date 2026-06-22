from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from database.models import GenderEnum, ScanTypeEnum

# ========== USER SCHEMAS ==========

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: Optional[str] = "doctor"

class UserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

# ========== PATIENT SCHEMAS ==========

class PatientCreate(BaseModel):
    full_name: str
    age: int
    gender: GenderEnum
    phone: Optional[str] = None
    notes: Optional[str] = None

class PatientResponse(BaseModel):
    id: int
    full_name: str
    age: int
    gender: GenderEnum
    phone: Optional[str]
    notes: Optional[str]
    doctor_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# ========== SCAN RESULT SCHEMAS ==========

class ScanResultCreate(BaseModel):
    scan_type: ScanTypeEnum
    prediction: str
    confidence: float
    all_probs: Optional[str] = None
    gradcam_image: Optional[str] = None
    input_data: Optional[str] = None
    patient_id: int

class ScanResultResponse(BaseModel):
    id: int
    scan_type: ScanTypeEnum
    prediction: str
    confidence: float
    all_probs: Optional[str]
    input_data: Optional[str]
    patient_id: int
    created_at: datetime

    class Config:
        from_attributes = True
