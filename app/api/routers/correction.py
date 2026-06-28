from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session

import Database.models as models_db
from Database.config import get_db
from .auth import get_current_user

router = APIRouter(prefix="/correction", tags=["Prediction Correction"])


@router.put("/{prediction_id}")
def correct_prediction(
    prediction_id: int,
    corrected_result: str = Form(...),
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user)
):
    # 1. Only doctors allowed
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Only doctors can correct predictions"
        )

    # 2. Get prediction
    prediction = db.query(models_db.Prediction).filter(
        models_db.Prediction.prediction_id == prediction_id
    ).first()

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    # 3. Update correction
    prediction.actual_result = corrected_result
    prediction.corrected_by_doctor_id = current_user.user_id

    db.commit()

    return {
        "prediction_id": prediction_id,
        "corrected_result": corrected_result,
        "status": "updated_successfully"
    }