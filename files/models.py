from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.database import Base
import enum

# ========== ENUMS ==========

class GenderEnum(str, enum.Enum):
    male = "Male"
    female = "Female"

class ScanTypeEnum(str, enum.Enum):
    chest_xray = "Chest X-Ray"
    brain_tumor = "Brain Tumor"
    brain_stroke = "Brain Stroke"
    diabetic_retinopathy = "Diabetic Retinopathy"
    ckd = "CKD"
    diabetes = "Diabetes"

# ========== TABLES ==========

class User(Base):
    """الطبيب أو المستخدم اللي بيدخل على النظام"""
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    full_name     = Column(String(100), nullable=False)
    email         = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role          = Column(String(50), default="doctor")  # doctor / admin
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    # العلاقات
    patients      = relationship("Patient", back_populates="doctor")


class Patient(Base):
    """بيانات المريض"""
    __tablename__ = "patients"

    id            = Column(Integer, primary_key=True, index=True)
    full_name     = Column(String(100), nullable=False)
    age           = Column(Integer, nullable=False)
    gender        = Column(Enum(GenderEnum), nullable=False)
    phone         = Column(String(20), nullable=True)
    notes         = Column(Text, nullable=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    # Foreign Key - كل مريض تابع لدكتور
    doctor_id     = Column(Integer, ForeignKey("users.id"), nullable=False)

    # العلاقات
    doctor        = relationship("User", back_populates="patients")
    scan_results  = relationship("ScanResult", back_populates="patient", cascade="all, delete-orphan")


class ScanResult(Base):
    """نتيجة كل تحليل - بيغطي الـ 6 models"""
    __tablename__ = "scan_results"

    id            = Column(Integer, primary_key=True, index=True)
    scan_type     = Column(Enum(ScanTypeEnum), nullable=False)
    prediction    = Column(String(100), nullable=False)   # النتيجة: Pneumonia / Normal / إلخ
    confidence    = Column(Float, nullable=False)          # نسبة الثقة
    all_probs     = Column(Text, nullable=True)            # JSON string للـ probabilities الكاملة
    gradcam_image = Column(Text, nullable=True)            # base64 image (للـ image models)
    input_data    = Column(Text, nullable=True)            # JSON string للـ input (للـ CKD/Diabetes)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    # Foreign Key
    patient_id    = Column(Integer, ForeignKey("patients.id"), nullable=False)

    # العلاقة
    patient       = relationship("Patient", back_populates="scan_results")
