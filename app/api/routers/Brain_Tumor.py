from fastapi import APIRouter, File, UploadFile, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

router = APIRouter(
    prefix="/Brain_MRI",
    tags=["Brain Tumor Analysis"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tumor_class_names = ['meningioma_tumor', 'no_tumor', 'glioma_tumor', 'pituitary_tumor']

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
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        raise RuntimeError(f"Brain Tumor model weights not found at '{path}'.")

tumor_model = load_tumor_model()

tumor_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@router.post("/predict")
async def predict_brain_tumor(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid format. Please upload a PNG or JPG Brain MRI image.")
        
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = tumor_preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = tumor_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, preds = torch.max(probabilities, dim=0)
        
        return {
            "filename": file.filename,
            "prediction": tumor_class_names[preds.item()],
            "confidence": round(conf.item(), 4),
            "all_probabilities": {tumor_class_names[i]: round(probabilities[i].item(), 4) for i in range(len(tumor_class_names))},
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))