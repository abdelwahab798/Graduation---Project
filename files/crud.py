from sqlalchemy.orm import Session
from database import models, schemas
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ========== USER CRUD ==========

def create_user(db: Session, user: schemas.UserCreate):
    hashed = pwd_context.hash(user.password)
    db_user = models.User(
        full_name=user.full_name,
        email=user.email,
        hashed_password=hashed,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

# ========== PATIENT CRUD ==========

def create_patient(db: Session, patient: schemas.PatientCreate, doctor_id: int):
    db_patient = models.Patient(**patient.model_dump(), doctor_id=doctor_id)
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

def get_patients_by_doctor(db: Session, doctor_id: int):
    return db.query(models.Patient).filter(models.Patient.doctor_id == doctor_id).all()

def get_patient_by_id(db: Session, patient_id: int):
    return db.query(models.Patient).filter(models.Patient.id == patient_id).first()

def delete_patient(db: Session, patient_id: int):
    patient = get_patient_by_id(db, patient_id)
    if patient:
        db.delete(patient)
        db.commit()
    return patient

# ========== SCAN RESULT CRUD ==========

def save_scan_result(db: Session, result: schemas.ScanResultCreate):
    db_result = models.ScanResult(**result.model_dump())
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    return db_result

def get_scans_by_patient(db: Session, patient_id: int):
    return db.query(models.ScanResult).filter(models.ScanResult.patient_id == patient_id).all()
