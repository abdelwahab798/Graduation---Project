from fastapi import HTTPException, APIRouter
from pydantic import BaseModel, Field
import joblib
import pandas as pd

router = APIRouter(
    prefix="/Diabetes",
    tags=["Diabetes_pipeline"]
)


try:
    lite_pipeline = joblib.load(r"D:\__Projects\Graduation---Project\app\models\diabetes_pipeline.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    lite_pipeline = None

class DiabetesPatientData(BaseModel):
    gender: str = Field(..., description="الجنس (Male / Female)", example="Male")
    age: float = Field(..., description="العمر بالسنوات", example=45.0)
    hypertension: int = Field(..., description="ضغط الدم المرتفع (1 لو موجود، 0 لو غير موجود)", example=0)
    heart_disease: int = Field(..., description="أمراض القلب (1 لو موجود، 0 لو غير موجود)", example=0)
    smoking_history: str = Field(..., description="تاريخ التدخين (مثال: never, current, former, No Info)", example="never")
    bmi: float = Field(..., description="مؤشر كتلة الجسم Body Mass Index", example=27.3)
    HbA1c_level: float = Field(..., description="مستوى السكر التراكمي في الدم", example=5.7)
    blood_glucose_level: float = Field(..., description="مستوى السكر العشوائي في الدم", example=130.0)

@router.post("/predict")
def predict_diabetes(data: DiabetesPatientData):
    if lite_pipeline is None:
        raise HTTPException(status_code=500, detail="Diabetes model pipeline is not loaded.")
    
    try:
       
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
       
        prediction = lite_pipeline.predict(input_df)[0]
        probabilities = lite_pipeline.predict_proba(input_df)[0]
        
     
        confidence = float(probabilities[prediction])
        
        return {
            "diabetes_prediction": int(prediction),  
            "probability": round(confidence * 100, 2),  
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")