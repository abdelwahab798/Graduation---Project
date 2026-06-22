from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.database import get_db
from database import crud, schemas
from database.auth import create_access_token, get_current_user
from database.models import User

# ─────────────────────────────────────────────
# AUTH ROUTER
# ─────────────────────────────────────────────

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

@auth_router.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = crud.get_user_by_email(db, user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db, user)

@auth_router.post("/login", response_model=schemas.TokenResponse)
def login(data: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, data.email)
    if not user or not crud.verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": user.email})
    return {"access_token": token}


# ─────────────────────────────────────────────
# PATIENTS ROUTER
# ─────────────────────────────────────────────

patients_router = APIRouter(prefix="/patients", tags=["Patients"])

@patients_router.post("/", response_model=schemas.PatientResponse)
def add_patient(
    patient: schemas.PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return crud.create_patient(db, patient, doctor_id=current_user.id)

@patients_router.get("/", response_model=list[schemas.PatientResponse])
def list_patients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return crud.get_patients_by_doctor(db, doctor_id=current_user.id)

@patients_router.get("/{patient_id}", response_model=schemas.PatientResponse)
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = crud.get_patient_by_id(db, patient_id)
    if not patient or patient.doctor_id != current_user.id:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@patients_router.delete("/{patient_id}")
def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = crud.get_patient_by_id(db, patient_id)
    if not patient or patient.doctor_id != current_user.id:
        raise HTTPException(status_code=404, detail="Patient not found")
    crud.delete_patient(db, patient_id)
    return {"message": "Patient deleted successfully"}


# ─────────────────────────────────────────────
# SCAN RESULTS ROUTER
# ─────────────────────────────────────────────

scans_router = APIRouter(prefix="/scans", tags=["Scan Results"])

@scans_router.get("/patient/{patient_id}", response_model=list[schemas.ScanResultResponse])
def get_patient_scans(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = crud.get_patient_by_id(db, patient_id)
    if not patient or patient.doctor_id != current_user.id:
        raise HTTPException(status_code=404, detail="Patient not found")
    return crud.get_scans_by_patient(db, patient_id)

@scans_router.get("/all", response_model=list[schemas.ScanResultResponse])
def get_all_my_scans(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return crud.get_all_scans_by_doctor(db, doctor_id=current_user.id)
