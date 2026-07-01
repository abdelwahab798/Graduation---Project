import io
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Depends,
    Form
)
from PIL import Image
from torchvision import models, transforms
from sqlalchemy.orm import Session
from typing import Optional
import os

from Database.config import get_db
import Database.models as models_db
from .auth import get_current_user

router = APIRouter(
    prefix="/Diabetic_Retinopathy",
    tags=["Diabetic_Retinopathy Analysis"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def register_state_hooks(model):
    model.gradients = None
    model.activations = None

    def forward_hook(module, input, output):
        model.activations = output

    def backward_hook(module, grad_input, grad_output):
        model.gradients = grad_output[0]

    model.layer4.register_forward_hook(forward_hook)
    model.layer4.register_full_backward_hook(backward_hook)


def load_retina_model():
    path = os.path.join(
        "models",
        "resnet18_diabetic_retinopathy_binary.pth"
    )

    try:
        model = models.resnet18()

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 2)
        )

        state_dict = torch.load(
            path,
            map_location=device
        )

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        # حقن الـ Hooks الآمنة
        register_state_hooks(model)

        return model

    except FileNotFoundError:
        raise RuntimeError(
            f"Weights not found at '{path}'."
        )


retina_model = load_retina_model()

retina_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@router.post("/predict")
async def predict_and_return_gradcam(
    file: UploadFile = File(...),
    patient_name: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user)
):

    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(
            status_code=403,
            detail="غير مسموح لك بإجراء هذا الفحص."
        )

    target_patient_id = None

    if current_user.role == "patient":

        if not current_user.patient_profile:
            raise HTTPException(
                status_code=400,
                detail="الملف الطبي للمريض غير مكتمل."
            )

        target_patient_id = (
            current_user.patient_profile.patient_id
        )

    else:
        target_patient_id = None

    if not file.filename.lower().endswith(
        ('.png', '.jpg', '.jpeg')
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid format. Please upload a PNG or JPG image."
        )

    try:
        image_data = await file.read()

        raw_image = Image.open(
            io.BytesIO(image_data)
        ).convert('RGB')

        input_tensor = (
            retina_preprocess(raw_image)
            .unsqueeze(0)
            .to(device)
        )

        output = retina_model(input_tensor)

        _, preds = output.max(1)

        predicted_class = preds.item()

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        confidence = probabilities[
            predicted_class
        ].item()

        class_names = [
            "No_DR",
            "DR"
        ]

        retina_model.zero_grad()

        output[0, predicted_class].backward()

        current_grads = (
            retina_model.gradients.detach()
        )

        current_acts = (
            retina_model.activations.detach()
        )

        pooled_gradients = torch.mean(
            current_grads,
            dim=[0, 2, 3]
        )

        heatmap = torch.zeros(
            current_acts.shape[2:],
            dtype=torch.float32,
            device=device
        )

        for i in range(current_acts.size(1)):
            heatmap += (
                pooled_gradients[i]
                * current_acts[0, i, :, :]
            )

        heatmap = heatmap.cpu().numpy()

        heatmap = np.maximum(
            heatmap,
            0
        )

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        img_cv = cv2.cvtColor(
            np.array(raw_image),
            cv2.COLOR_RGB2BGR
        )

        img_cv = cv2.resize(
            img_cv,
            (224, 224)
        )

        heatmap_resized = cv2.resize(
            heatmap,
            (224, 224)
        )

        heatmap_resized = np.uint8(
            255 * heatmap_resized
        )

        heatmap_colored = cv2.applyColorMap(
            heatmap_resized,
            cv2.COLORMAP_JET
        )

        superimposed_img = cv2.addWeighted(
            img_cv,
            0.6,
            heatmap_colored,
            0.4,
            0
        )

        _, encoded_img = cv2.imencode(
            '.png',
            superimposed_img
        )

        base64_img = base64.b64encode(
            encoded_img.tobytes()
        ).decode('utf-8')

        gradcam_base64_url = (
            f"data:image/png;base64,{base64_img}"
        )

      
        os.makedirs(
            "static/uploads/eyes",
            exist_ok=True
        )

        rand_id = int(
            torch.randint(
                0,
                10000,
                (1,)
            ).item()
        )

        saved_image_path = (
            f"static/uploads/eyes/"
            f"{current_user.user_id}_"
            f"{rand_id}_"
            f"{file.filename}"
        )

        with open(
            saved_image_path,
            "wb"
        ) as f:
            f.write(image_data)

        db_exam = models_db.ImageExamination(
            patient_id=target_patient_id,
            patient_name_note=patient_name,
            doctor_id=current_user.user_id if current_user.role == "doctor" else None,
            modality="Retinopathy_Image",
            image_path=saved_image_path
        )

        db.add(db_exam)
        db.flush()

        db_prediction = models_db.Prediction(
            patient_id=target_patient_id,
            exam_id=db_exam.exam_id,
            model_name="ResNet18_DiabeticRetinopathy_v1",
            prediction_result=class_names[predicted_class],
            confidence_score=round(
                confidence,
                4
            ),
            gradcam_path=gradcam_base64_url,
            actual_result=None
        )

        db.add(db_prediction)

        db.commit()

        return {
            "prediction_id":
                db_prediction.prediction_id,

            "filename":
                file.filename,

            "prediction":
                class_names[predicted_class],

            "confidence":
                round(
                    confidence,
                    4
                ),

            "status":
                "success",

            "gradcam_image":
                gradcam_base64_url
        }

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )