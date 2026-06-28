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
from torchvision import models, transforms
from PIL import Image
from sqlalchemy.orm import Session
import os
from typing import Optional

from Database.config import get_db
import Database.models as models_db
from .auth import get_current_user

router = APIRouter(
    prefix="/Brain_Stroke",
    tags=["Brain Stroke Analysis"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stroke_class_names = [
    "Normal",
    "Ischemia",
    "Bleeding"
]


def register_state_hooks(model):
    model.gradients = None
    model.activations = None

    def forward_hook(module, input, output):
        model.activations = output

    def backward_hook(module, grad_input, grad_output):
        model.gradients = grad_output[0]

    model.layer4.register_forward_hook(forward_hook)
    model.layer4.register_full_backward_hook(backward_hook)


def process_mobile_photo_for_stroke(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise ValueError("Invalid image file")

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        img_enhanced = clahe.apply(img)

        img_resized = cv2.resize(
            img_enhanced,
            (224, 224),
            interpolation=cv2.INTER_CUBIC
        )

        return img_resized

    except Exception as e:
        raise ValueError(
            f"Image preprocessing error: {str(e)}"
        )


def load_stroke_model():
    path = "models/stroke_resnet18.pth"

    try:
        model = models.resnet18()

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 3)

        state_dict = torch.load(
            path,
            map_location=device
        )

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        register_state_hooks(model)

        return model

    except FileNotFoundError:
        raise RuntimeError(
            f"Stroke model not found at '{path}'."
        )


stroke_model = load_stroke_model()

stroke_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(
        lambda x: x.repeat(3, 1, 1)
    ),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@router.post("/predict")
async def predict_brain_stroke(
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
        (".png", ".jpg", ".jpeg")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG and JPEG files are allowed."
        )

    try:
        image_data = await file.read()

        processed_array = process_mobile_photo_for_stroke(
            image_data
        )

        pil_image = Image.fromarray(processed_array)

        input_tensor = (
            stroke_preprocess(pil_image)
            .unsqueeze(0)
            .to(device)
        )

        outputs = stroke_model(input_tensor)

        probabilities = torch.softmax(
            outputs[0],
            dim=0
        )

        confidence, prediction_idx = torch.max(
            probabilities,
            dim=0
        )

        prediction = stroke_class_names[
            prediction_idx.item()
        ]

        stroke_model.zero_grad()

        outputs[0][prediction_idx].backward()

        current_grads = (
            stroke_model.gradients.detach()
        )

        current_acts = (
            stroke_model.activations.detach()
        )

        pooled_gradients = torch.mean(
            current_grads,
            dim=[0,2, 3]
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

        heatmap[heatmap < 0.5] = 0

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        img_cv = cv2.cvtColor(
            processed_array,
            cv2.COLOR_GRAY2BGR
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
            ".png",
            superimposed_img
        )

        base64_img = base64.b64encode(
            encoded_img.tobytes()
        ).decode("utf-8")

        gradcam_base64_url = (
            f"data:image/png;base64,{base64_img}"
        )

        os.makedirs(
            "static/uploads",
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
            f"static/uploads/"
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
            modality="CT",
            doctor_id=current_user.user_id if current_user.role == "doctor" else None,
            image_path=saved_image_path
        )

        db.add(db_exam)
        db.flush()

        db_prediction = models_db.Prediction(
            patient_id=target_patient_id,
            exam_id=db_exam.exam_id,
            model_name="ResNet18_BrainStroke_v1",
            prediction_result=prediction,
            confidence_score=round(
                confidence.item(),
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
                prediction,

            "confidence":
                round(
                    confidence.item(),
                    4
                ),

            "all_probabilities": {
                stroke_class_names[i]:
                round(
                    probabilities[i].item(),
                    4
                )
                for i in range(
                    len(stroke_class_names)
                )
            },

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