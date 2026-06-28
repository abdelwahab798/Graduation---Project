from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from Database.config import get_db
import Database.models as models_db
from .auth import get_current_user

router = APIRouter(
    prefix="/medical-records",
    tags=["Medical Records & History"]
)

@router.get("/dashboard-history")
async def get_dashboard_history(
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user)
):
    """
    جلب السجل الطبي الشامل (أشعة + تحاليل كلى + سكر) بناءً على صلاحية المستخدم
    """
    try:
        if current_user.role == "patient":
            if not current_user.patient_profile:
                raise HTTPException(status_code=400, detail="الملف الطبي للمريض غير مكتمل.")
            p_id = current_user.patient_profile.patient_id

            xrays    = db.query(models_db.ImageExamination).filter(models_db.ImageExamination.patient_id == p_id).all()
            ckd      = db.query(models_db.CKDData).filter(models_db.CKDData.patient_id == p_id).all()
            diabetes = db.query(models_db.DiabetesData).filter(models_db.DiabetesData.patient_id == p_id).all()

            # بيانات المريض من Patient profile
            patient_age    = current_user.patient_profile.age
            patient_gender = current_user.patient_profile.gender

        else:
            d_id     = current_user.user_id
            xrays    = db.query(models_db.ImageExamination).filter(models_db.ImageExamination.doctor_id == d_id).all()
            ckd      = db.query(models_db.CKDData).filter(models_db.CKDData.doctor_id == d_id).all()
            diabetes = db.query(models_db.DiabetesData).filter(models_db.DiabetesData.doctor_id == d_id).all()

            patient_age    = None
            patient_gender = None

        return {
            "status": "success",
            "records": {

                # ── أشعة ──────────────────────────────────────────────────────────
                "xray_examinations": [
                    {
                        "exam_id":       x.exam_id,
                        "patient_name":  x.patient_name_note if (current_user.role == "doctor" and x.patient_name_note) else current_user.full_name,
                        "modality":      x.modality,
                        "image_path":    x.image_path,
                        "date":          x.exam_date.strftime("%Y-%m-%d %H:%M"),
                        "prediction":    x.predictions[0].prediction_result if x.predictions else "No prediction",
                        "actual_result": x.predictions[0].actual_result     if x.predictions else None,
                        "prediction_id": x.predictions[0].prediction_id     if x.predictions else None,
                        "confidence":    x.predictions[0].confidence_score  if x.predictions else None,
                        "gradcam":       x.predictions[0].gradcam_path      if x.predictions else None,
                    } for x in xrays
                ],

                # ── CKD ───────────────────────────────────────────────────────────
                "ckd_records": [
                    {
                        "record_id":        c.record_id,
                        "patient_name":     c.patient_name_note if (current_user.role == "doctor" and c.patient_name_note) else current_user.full_name,
                        "date":             c.recorded_at.strftime("%Y-%m-%d %H:%M"),
                        "prediction":       c.predictions[0].prediction_result if c.predictions else "No prediction",
                        "actual_result":    c.predictions[0].actual_result     if c.predictions else None,
                        "prediction_id":    c.predictions[0].prediction_id     if c.predictions else None,
                        "confidence":       c.predictions[0].confidence_score  if c.predictions else None,
                        # ── القيم المخبرية ──
                        "gfr":              c.gfr,
                        "serum_creatinine": c.serum_creatinine,
                        "bun":              c.bun,
                        "blood_pressure":   c.blood_pressure,
                        "urine_ph":         c.urine_ph,
                        "oxalate_levels":   c.oxalate_levels,
                        "c3_c4":            c.c3_c4,
                        "months":           c.months,
                        "stress_level":     c.stress_level,
                        "family_history":   c.family_history,
                    } for c in ckd
                ],

                # ── Diabetes ──────────────────────────────────────────────────────
                "diabetes_records": [
                    {
                        "record_id":          d.record_id,
                        "patient_name":       d.patient_name_note if (current_user.role == "doctor" and d.patient_name_note) else current_user.full_name,
                        "date":               d.recorded_at.strftime("%Y-%m-%d %H:%M"),
                        "prediction":         d.predictions[0].prediction_result if d.predictions else "No prediction",
                        "actual_result":      d.predictions[0].actual_result     if d.predictions else None,
                        "prediction_id":      d.predictions[0].prediction_id     if d.predictions else None,
                        "confidence":         d.predictions[0].confidence_score  if d.predictions else None,
                        # ── البيانات الصحية ──
                        # gender & age: من Patient profile لو مريض، أو None لو دكتور (مش موجودة في DiabetesData)
                        "gender":             patient_gender,
                        "age":                patient_age,
                        "hypertension":       d.hypertension,
                        "heart_disease":      d.heart_disease,
                        "smoking_history":    d.smoking_history,
                        "bmi":                d.bmi,
                        "hba1c":              d.hba1c_level,
                        "blood_glucose_level": d.blood_glucose_level,
                    } for d in diabetes
                ],
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))