import io
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
from torchvision import models, transforms

router = APIRouter(
    prefix="/Diabetic_Retinopathy",
    tags=["Diabetic_Retinopathy Analysis"]
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

def load_retina_model():
    path = r"models\resnet18_diabetic_retinopathy_binary.pth"
    try:
        state_dict = torch.load(path, map_location=device)
        
        model = models.resnet18()
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 2)
        )
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        model.layer4.register_forward_hook(forward_hook)
        model.layer4.register_full_backward_hook(backward_hook)
        
        return model
    except FileNotFoundError:
        raise RuntimeError(f"Weights not found at '{path}'.")

retina_model = load_retina_model()

retina_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.post("/predict")
async def predict_and_return_gradcam(file: UploadFile = File(...)):
    global gradients, activations
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid format. Please upload a PNG or JPG image.")
        
    try:
        image_data = await file.read()
        raw_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        input_tensor = retina_preprocess(raw_image).unsqueeze(0).to(device)
        
        output = retina_model(input_tensor)
        _, preds = output.max(1)
        predicted_class = preds.item()
        
        probabilities = torch.softmax(output, dim=1)[0]
        confidence = probabilities[predicted_class].item()
        class_names = ["No_DR", "DR"]
        
        retina_model.zero_grad()
        output[0, predicted_class].backward()
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        img_cv = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (224, 224))
        
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
        
       
        _, encoded_img = cv2.imencode('.png', superimposed_img)
        
        
        base64_img = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
        
        return {
            "filename": file.filename,
            "prediction": class_names[predicted_class],
            "confidence": round(confidence, 4),
            "status": "success",
            "gradcam_image": f"data:image/png;base64,{base64_img}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))