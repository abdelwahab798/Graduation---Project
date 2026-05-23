from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import cv2

router = APIRouter(
    prefix="/Brain_Stroke",
    tags=["Brain Stroke Analysis"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stroke_class_names = ['Normal', 'Ischemia', 'Bleeding']

def process_mobile_photo_for_stroke(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Invalid image file")
            
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img)
        
        img_resized = cv2.resize(img_enhanced, (224, 224), interpolation=cv2.INTER_CUBIC)
        return img_resized
    except Exception as e:
        raise ValueError(f"Mobile Image Preprocessing Error: {str(e)}")

def load_stroke_model():
    path = r"D:\__Projects\Graduation---Project\app\models\stroke_resnet18.pth" 
    model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 3)
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        raise RuntimeError(f"Stroke model weights not found at '{path}'.")

stroke_model = load_stroke_model()

stroke_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.post("/predict")
async def predict_brain_stroke(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400, 
            detail="MediScan AI accepts PNG, JPG, or JPEG photos of CT scans for this version."
        )
        
    try:
        file_bytes = await file.read()
        processed_array = process_mobile_photo_for_stroke(file_bytes)
        pil_img = Image.fromarray(processed_array)
        
        input_tensor = stroke_preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = stroke_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, preds = torch.max(probabilities, dim=0)
            
        return {
            "filename": file.filename,
            "prediction": stroke_class_names[preds.item()],
            "confidence": round(conf.item(), 4),
            "all_probabilities": {stroke_class_names[i]: round(probabilities[i].item(), 4) for i in range(len(stroke_class_names))},
            "status": "success",
            "ui_notice": "For maximum accuracy, ensure the photo is well-lit, centered, and covers the entire CT scan frame."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")