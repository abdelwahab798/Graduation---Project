import datetime
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from .config import Base

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(String, default="patient", nullable=False) 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    patient_profile = relationship("Patient", back_populates="user", uselist=False, cascade="all, delete")
    
    corrected_predictions = relationship(
        "Prediction", 
        back_populates="doctor",
        foreign_keys="[Prediction.corrected_by_doctor_id]"
    )


class Patient(Base):
    __tablename__ = "patients"
    
    patient_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), unique=True)
    
    age = Column(Float, nullable=False) 
    gender = Column(String, nullable=False) 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="patient_profile")
    images = relationship("ImageExamination", back_populates="patient", cascade="all, delete")
    ckd_records = relationship("CKDData", back_populates="patient", cascade="all, delete")
    diabetes_records = relationship("DiabetesData", back_populates="patient", cascade="all, delete")
    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete")


class ImageExamination(Base):
    __tablename__ = "image_examinations"
    
    exam_id = Column(Integer, primary_key=True, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=True)
    
    doctor_id = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    
    patient_name_note = Column(String, nullable=True)
    
    modality = Column(String) 
    image_path = Column(String, nullable=False) 
    exam_date = Column(DateTime, default=datetime.datetime.utcnow)
    
   
    patient = relationship("Patient", back_populates="images")
    predictions = relationship("Prediction", back_populates="image_exam")
    doctor = relationship("User", foreign_keys=[doctor_id])


class CKDData(Base):
    __tablename__ = "ckd_data"
    
    record_id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id", ondelete="CASCADE"),nullable=True)
    doctor_id = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    patient_name_note = Column(String, nullable=True)
    gfr = Column(Float, nullable=False)
    c3_c4 = Column(Float, nullable=False)
    bun = Column(Float, nullable=False)
    blood_pressure = Column(Float, nullable=False)
    serum_creatinine = Column(Float, nullable=False)
    urine_ph = Column(Float, nullable=False)
    months = Column(Float, nullable=False)
    oxalate_levels = Column(Float, nullable=False)
    stress_level = Column(String, nullable=False)
    family_history = Column(String, nullable=False)
    
    recorded_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    patient = relationship("Patient", back_populates="ckd_records")
    predictions = relationship("Prediction", back_populates="ckd_record")


class DiabetesData(Base):
    __tablename__ = "diabetes_data"
    
    record_id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id", ondelete="CASCADE"),nullable=True)
    doctor_id = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    patient_name_note = Column(String, nullable=True)
    hypertension = Column(Integer, nullable=False)
    heart_disease = Column(Integer, nullable=False)
    smoking_history = Column(String, nullable=False)
    bmi = Column(Float, nullable=False)
    hba1c_level = Column(Float, nullable=False)
    blood_glucose_level = Column(Float, nullable=False)
    
    recorded_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    patient = relationship("Patient", back_populates="diabetes_records")
    predictions = relationship("Prediction", back_populates="diabetes_record")


class Prediction(Base):
    __tablename__ = "predictions"
    
    prediction_id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=True)
    
    doctor_id = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    
    exam_id = Column(Integer, ForeignKey("image_examinations.exam_id", ondelete="SET NULL"), nullable=True)
    ckd_record_id = Column(Integer, ForeignKey("ckd_data.record_id", ondelete="SET NULL"), nullable=True)
    diabetes_record_id = Column(Integer, ForeignKey("diabetes_data.record_id", ondelete="SET NULL"), nullable=True)
    
    model_name = Column(String, nullable=False)        
    prediction_result = Column(String, nullable=False) 
    confidence_score = Column(Float, nullable=True)    
    gradcam_path = Column(String, nullable=True) 
    
    actual_result = Column(String, default=None) 
    corrected_by_doctor_id = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    
    predicted_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    patient = relationship("Patient", back_populates="predictions")
    image_exam = relationship("ImageExamination", back_populates="predictions")
    ckd_record = relationship("CKDData", back_populates="predictions")
    diabetes_record = relationship("DiabetesData", back_populates="predictions")
    
    exam_doctor = relationship("User", foreign_keys=[doctor_id])
    doctor = relationship("User", foreign_keys=[corrected_by_doctor_id], back_populates="corrected_predictions")