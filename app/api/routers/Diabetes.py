from fastapi import HTTPException, APIRouter, Depends
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from sqlalchemy.orm import Session

from Database.config import get_db
import Database.models as models_db
from .auth import get_current_user

router = APIRouter(prefix="/Diabetes", tags=["Diabetes_pipeline"])

try:
    lite_pipeline = joblib.load(r"D:\__Projects\Graduation---Project\app\models\diabetes_pipeline.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    lite_pipeline = None

class DiabetesPatientData(BaseModel):
    patient_name_note: str | None = None
    gender: str = Field(..., example="Male")
    age: float = Field(..., example=45.0)
    hypertension: int = Field(..., example=0)
    heart_disease: int = Field(..., example=0)
    smoking_history: str = Field(..., example="never")
    bmi: float = Field(..., example=27.3)
    HbA1c_level: float = Field(..., example=5.7)
    blood_glucose_level: float = Field(..., example=130.0)

@router.post("/predict")
def predict_diabetes(
    data: DiabetesPatientData,
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user)
):
    if lite_pipeline is None:
        raise HTTPException(status_code=500, detail="Diabetes model pipeline is not loaded.")

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

        # DB logging
        db_record = models_db.DiabetesData(
            patient_id=patient_id,
            doctor_id=doctor_id,
            patient_name_note=data.patient_name_note if current_user.role == "doctor" else None, 
            hypertension=data.hypertension,
            heart_disease=data.heart_disease,
            smoking_history=data.smoking_history,
            bmi=data.bmi,
            hba1c_level=data.HbA1c_level,
            blood_glucose_level=data.blood_glucose_level
        )
        db.add(db_record)
        db.flush()

        db_prediction = models_db.Prediction(
            patient_id=patient_id,
            doctor_id=doctor_id,
            diabetes_record_id=db_record.record_id,
            model_name="Diabetes_Pipeline_v1",
            prediction_result=str(prediction),
            confidence_score=round(confidence, 4),
        )
        db.add(db_prediction)
        db.commit()

        return {
            "prediction_id": db_prediction.prediction_id,
            "diabetes_prediction": prediction,
            "probability": round(confidence * 100, 2),
            "status": "Success",
            "can_correct": current_user.role == "doctor"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")