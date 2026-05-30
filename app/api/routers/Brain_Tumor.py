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
    prefix="/Brain_MRI",
    tags=["Brain Tumor Analysis"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tumor_class_names = ['meningioma_tumor', 'no_tumor', 'glioma_tumor', 'pituitary_tumor']

gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

def load_tumor_model():
    path = "models/resnet18_brain_tumor.pth"
    model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 4)  )
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        model.layer4.register_forward_hook(forward_hook)
        model.layer4.register_full_backward_hook(backward_hook)
        
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
    global gradients, activations
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid format. Please upload a PNG or JPG Brain MRI image.")
        
    try:
        image_data = await file.read()
        raw_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        input_tensor = tumor_preprocess(raw_image).unsqueeze(0).to(device)
        
        
        outputs = tumor_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, preds = torch.max(probabilities, dim=0)
        
       
        tumor_model.zero_grad()
        score = outputs[0][preds]
        score.backward()
        
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
            "prediction": tumor_class_names[preds.item()],
            "confidence": round(conf.item(), 4),
            "all_probabilities": {tumor_class_names[i]: round(probabilities[i].item(), 4) for i in range(len(tumor_class_names))},
            "status": "success",
            "gradcam_image": f"data:image/png;base64,{base64_img}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))