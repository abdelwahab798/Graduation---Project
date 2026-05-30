import io
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, File, UploadFile, HTTPException
from torchvision import models, transforms
from PIL import Image

router = APIRouter(
    prefix="/chest-xray",
    tags=["Chest X-Ray Analysis"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

def load_chest_model():
    path = "models/chest_xray_best_model.pth"
    try:
        checkpoint = torch.load(path, map_location=device)
        model = models.resnet18()
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)  # مخرج واحد (Binary)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        model.layer4.register_forward_hook(forward_hook)
        model.layer4.register_full_backward_hook(backward_hook)
        
        return model, checkpoint
    except FileNotFoundError:
        raise RuntimeError(f"Chest X-Ray model checkpoint not found at '{path}'.")

chest_model, chest_info = load_chest_model()

img_size = chest_info["image_size"]
chest_preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(chest_info["normalization_mean"], chest_info["normalization_std"])
])

@router.post("/predict")
async def predict_chest_xray(file: UploadFile = File(...)):
    global gradients, activations
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid format. Please upload a PNG or JPG Chest X-Ray image.")
        
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
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        img_cv = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (img_size, img_size))
        
        heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
        
        _, encoded_img = cv2.imencode('.png', superimposed_img)
        base64_img = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
        
        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probability": round(prob, 4),
            "status": "success",
            "gradcam_image": f"data:image/png;base64,{base64_img}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))