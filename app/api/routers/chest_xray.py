from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

router = APIRouter(
    prefix="/chest-xray",
    tags=["Chest X-Ray Analysis"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return model, checkpoint
    except FileNotFoundError:
        raise RuntimeError(f"Chest X-Ray model checkpoint not found at '{path}'.")

chest_model, chest_info = load_chest_model()

chest_preprocess = transforms.Compose([
    transforms.Resize((chest_info["image_size"], chest_info["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(chest_info["normalization_mean"], chest_info["normalization_std"])
])

@router.post("/predict")
async def predict_chest_xray(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid format. Please upload a PNG or JPG Chest X-Ray image.")
        
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        input_tensor = chest_preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = chest_model(input_tensor)
            prob = torch.sigmoid(output).item()
            prediction = "Pneumonia" if prob > 0.5 else "Normal"
    
        return {
            "filename": file.filename,
            "prediction": prediction,
            "probability": round(prob, 4),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))