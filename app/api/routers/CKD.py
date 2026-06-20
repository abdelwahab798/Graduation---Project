from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

router=APIRouter(prefix="/CKD",
                 tags=["CKD_pipeline"])
try:
    lite_pipeline = joblib.load(r"D:\__Projects\Graduation---Project\app\models\ckd_stage_lite_pipeline.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    lite_pipeline = None

class PatientData(BaseModel):
    gfr: float = Field(..., description="Glomerular Filtration Rate", example=90.0)
    c3_c4: float = Field(..., description="C3/C4 Complement levels", example=1.2)
    bun: float = Field(..., description="Blood Urea Nitrogen", example=15.0)
    blood_pressure: float = Field(..., description="Blood Pressure (Systolic)", example=120.0)
    serum_creatinine: float = Field(..., description="Serum Creatinine level", example=1.0)
    urine_ph: float = Field(..., description="Urine pH level", example=6.0)
    months: float = Field(..., description="Duration of symptoms in months", example=6.0)
    oxalate_levels: float = Field(..., description="Oxalate Levels", example=20.0)
    stress_level: str = Field(..., description="مستوى التوتر (مثال: Low, Medium, High)", example="Low")
    family_history: str = Field(..., description="التاريخ العائلي: Yes أو No", example="No")

@router.post("/predict")
def predict_ckd_stage(data: PatientData):
    if lite_pipeline is None:
        raise HTTPException(status_code=500, detail="Lite model pipeline is not loaded.")
    
    try:
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        prediction = lite_pipeline.predict(input_df)[0]
        probabilities = lite_pipeline.predict_proba(input_df)[0]
        
        return {
            "predicted_stage": int(prediction),
            "confidence_score": round(float(probabilities[prediction]) * 100, 2), 
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

