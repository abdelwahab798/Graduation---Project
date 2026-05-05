from fastapi import APIRouter, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

router = APIRouter(
    prefix="/chest-xray",
    tags=["Chest X-Ray Analysis"]
)

# 1. Load Model Metadata (نفس اللي سيفتها في الـ checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    checkpoint = torch.load("models/chest_xray_best_model.pth", map_location=device)
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

model, info = load_model()

# 2. Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((info["image_size"], info["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(info["normalization_mean"], info["normalization_std"])
])

@router.post("/predict")
async def predict_chest_xray(file: UploadFile = File(...)):
    # قراءة الصورة
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # التحويل للـ Tensor
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "Pneumonia" if prob > 0.5 else "Normal"
    
    return {
        "filename": file.filename,
        "prediction": prediction,
        "probability": round(prob, 4),
        "status": "success"
    }