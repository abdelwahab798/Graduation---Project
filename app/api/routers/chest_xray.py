import io
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from torchvision import models, transforms
from PIL import Image
from sqlalchemy.orm import Session
import os
from typing import Optional

from Database.config import get_db
import Database.models as models_db
from .auth import get_current_user

router = APIRouter(
    prefix="/chest-xray",
    tags=["Chest X-Ray Analysis"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def blackout_edges(pil_img):
    img = np.array(pil_img)
    pad = 35
    img[:pad, :, :] = 0
    img[-pad:, :, :] = 0
    img[:, :pad, :] = 0
    img[:, -pad:, :] = 0
    return Image.fromarray(img)

def register_state_hooks(model):
    model.gradients = None
    model.activations = None

    def forward_hook(module, input, output):
        model.activations = output

    def backward_hook(module, grad_input, grad_output):
        model.gradients = grad_output[0]

    model.layer4.register_forward_hook(forward_hook)
    model.layer4.register_full_backward_hook(backward_hook)

def load_chest_model():
    path = "models/chest_xray_best_model.pth"
    try:
        checkpoint = torch.load(path, map_location=device)
        model = models.resnet18()
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        register_state_hooks(model)
        return model, checkpoint
    except FileNotFoundError:
        raise RuntimeError(f"Chest X-Ray model not found at '{path}'.")

chest_model, chest_info = load_chest_model()

img_size = chest_info["image_size"]
chest_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(blackout_edges),
    transforms.CenterCrop((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(chest_info["normalization_mean"], chest_info["normalization_std"])
])

@router.post("/predict")
async def predict_chest_xray(
    file: UploadFile = File(...),
    patient_name: Optional[str] = Form(None), # 👈 استقبال اسم المريض النصي المرسل من الـ FormData
    db: Session = Depends(get_db),
    current_user: models_db.User = Depends(get_current_user)
):
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="غير مسموح لك بإجراء هذا الفحص.")

    target_patient_id = None

    # ✅ تحديد الهوية بناءً على الصلاحية الحالية
    if current_user.role == "patient":
        if not current_user.patient_profile:
            raise HTTPException(status_code=400, detail="الملف الطبي للمريض غير مكتمل.")
        target_patient_id = current_user.patient_profile.patient_id
    else:
        # لو دكتور، الـ target_patient_id هيفضل None لأنه مجرد اسم تذكيري مش مربوط بـ ID في الداتابيز
        target_patient_id = None 

    try:
        image_data = await file.read()
        raw_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        input_tensor = chest_preprocess(raw_image).unsqueeze(0).to(device)

        output = chest_model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "Pneumonia" if prob > 0.5 else "Normal"
        confidence = prob if prob > 0.5 else (1.0 - prob)

        chest_model.zero_grad()
        output[0].backward()

        current_grads = chest_model.gradients.detach()
        current_acts = chest_model.activations.detach()

        pooled_gradients = torch.mean(current_grads, dim=[2, 3]).squeeze(0)

        heatmap = torch.zeros(current_acts.shape[2:], dtype=torch.float32, device=device)
        for i in range(current_acts.size(1)):
            heatmap += pooled_gradients[i] * current_acts[0, i, :, :]

        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        heatmap[heatmap < 0.5] = 0
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        processed_raw = blackout_edges(raw_image)
        img_cv = cv2.cvtColor(np.array(processed_raw), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (256, 256))

        start_x = (256 - img_size) // 2
        start_y = (256 - img_size) // 2
        img_cv_cropped = img_cv[start_y:start_y+img_size, start_x:start_x+img_size]

        heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(img_cv_cropped, 0.6, heatmap_colored, 0.4, 0)

        _, encoded_img = cv2.imencode('.png', superimposed_img)
        base64_img = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
        gradcam_base64_url = f"data:image/png;base64,{base64_img}"

        # ✅ حفظ الصورة
        os.makedirs("static/uploads", exist_ok=True)
        rand_id = int(torch.randint(0, 10000, (1,)).item())
        saved_image_path = f"static/uploads/{current_user.user_id}_{rand_id}_{file.filename}"
        with open(saved_image_path, "wb") as f:
            f.write(image_data)

        # ✅ حفظ سجل الفحص
        # ملاحظة: إذا كان جدول ImageExamination لا يقبل patient_id كـ Null، تأكد من تعديل الـ Schema في الداتابيز لتسمح بـ Nullable=True
        db_exam = models_db.ImageExamination(
            patient_id=target_patient_id,
            patient_name_note=patient_name, 
            modality="X-Ray",
            doctor_id=current_user.user_id if current_user.role == "doctor" else None,
            image_path=saved_image_path
        )
        db.add(db_exam)
        db.flush()

        # ✅ حفظ التنبؤ
        # تذكير: يمكنك إضافة حقل ملاحظات (مثل notes=patient_name) لتخزين الاسم التذكيري إذا أردت استرجاعه لاحقاً للـ Doctor التاريخ الخاص به
        db_prediction = models_db.Prediction(
            patient_id=target_patient_id,
            exam_id=db_exam.exam_id,
            model_name="ResNet18_ChestXray_v1",
            prediction_result=prediction,
            confidence_score=round(confidence, 4),
            gradcam_path=gradcam_base64_url,
            actual_result=None
        )
        db.add(db_prediction)
        db.commit()

        return {
            "prediction_id": db_prediction.prediction_id,
            "filename": file.filename,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probability": round(prob, 4),
            "status": "success",
            "gradcam_image": gradcam_base64_url
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))