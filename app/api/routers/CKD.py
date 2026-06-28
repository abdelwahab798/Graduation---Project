from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

from sqlalchemy.orm import Session
import Database.models as models_db
from Database.config import get_db
from .auth import get_current_user


router = APIRouter(prefix="/CKD", tags=["CKD_pipeline"])

# load model
try:
    lite_pipeline = joblib.load(
        r"D:\__Projects\Graduation---Project\app\models\ckd_stage_lite_pipeline.pkl"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    lite_pipeline = None


# input schema
class PatientData(BaseModel):
    patient_name_note: str | None = None
    gfr: float
    c3_c4: float
    bun: float
    blood_pressure: float
    serum_creatinine: float
    urine_ph: float
    months: float
    oxalate_levels: float
    stress_level: str
    family_history: str


@router.post("/predict")
def predict_ckd_stage(
    data: PatientData,
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user)
):
    if lite_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # تحديد patient_id و doctor_id
    patient_id = None
    doctor_id = None

    if current_user.role == "patient":
        if not current_user.patient_profile:
            raise HTTPException(status_code=403, detail="Patient profile not found")
        patient_id = current_user.patient_profile.patient_id

    elif current_user.role == "doctor":
        doctor_id = current_user.user_id

    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = int(lite_pipeline.predict(input_df)[0])
        confidence = float(lite_pipeline.predict_proba(input_df)[0][prediction])

        db_record = models_db.CKDData(
            patient_id=patient_id,  # None لو doctor
            doctor_id=doctor_id,    # None لو patient
            gfr=data.gfr,
            c3_c4=data.c3_c4,
            bun=data.bun,
            patient_name_note=data.patient_name_note if current_user.role == "doctor" else None, 
            blood_pressure=data.blood_pressure,
            serum_creatinine=data.serum_creatinine,
            urine_ph=data.urine_ph,
            months=data.months,
            oxalate_levels=data.oxalate_levels,
            stress_level=data.stress_level,
            family_history=data.family_history
        )
        db.add(db_record)
        db.flush()

        db_prediction = models_db.Prediction(
            patient_id=patient_id,
            doctor_id=doctor_id,
            ckd_record_id=db_record.record_id,
            model_name="CKD_Lite_Pipeline_v1",
            prediction_result=str(prediction),
            confidence_score=round(confidence, 4),
        )
        db.add(db_prediction)
        db.commit()

        return {
            "prediction_id": db_prediction.prediction_id,
            "predicted_stage": prediction,
            "confidence_score": round(confidence * 100, 2),
            "status": "Success",
            "can_correct": current_user.role == "doctor"  # ← الـ frontend يعرف يظهر الـ field ولا لأ
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))