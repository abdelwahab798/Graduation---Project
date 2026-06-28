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
    prefix="/Brain_MRI",
    tags=["Brain Tumor Analysis"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tumor_class_names = [
    'meningioma_tumor',
    'no_tumor',
    'glioma_tumor',
    'pituitary_tumor'
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


def load_tumor_model():
    path = "models/resnet18_brain_tumor.pth"

    model = models.resnet18()

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 4)
    )

    try:
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
            f"Brain Tumor model weights not found at '{path}'."
        )


tumor_model = load_tumor_model()

tumor_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


@router.post("/predict")
async def predict_brain_tumor(
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
            detail="Invalid format. Please upload a PNG or JPG Brain MRI image."
        )

    try:
        image_data = await file.read()

        raw_image = Image.open(
            io.BytesIO(image_data)
        ).convert('RGB')

        input_tensor = (
            tumor_preprocess(raw_image)
            .unsqueeze(0)
            .to(device)
        )

        outputs = tumor_model(input_tensor)

        probabilities = torch.nn.functional.softmax(
            outputs[0],
            dim=0
        )

        conf, preds = torch.max(
            probabilities,
            dim=0
        )

        prediction_class = tumor_class_names[
            preds.item()
        ]

        # GradCAM (بدون أي تعديل)
        tumor_model.zero_grad()

        score = outputs[0][preds]

        score.backward()

        current_grads = (
            tumor_model.gradients.detach()
        )

        current_acts = (
            tumor_model.activations.detach()
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
            modality="MRI",
            doctor_id=current_user.user_id if current_user.role == "doctor" else None,
            image_path=saved_image_path
        )

        db.add(db_exam)
        db.flush()

        db_prediction = models_db.Prediction(
            patient_id=target_patient_id,
            exam_id=db_exam.exam_id,
            model_name="ResNet18_BrainMRI_v1",
            prediction_result=prediction_class,
            confidence_score=round(
                conf.item(),
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
                prediction_class,

            "confidence":
                round(
                    conf.item(),
                    4
                ),

            "all_probabilities": {
                tumor_class_names[i]:
                round(
                    probabilities[i].item(),
                    4
                )
                for i in range(
                    len(tumor_class_names)
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
