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

def blackout_edges(pil_img):
    """ دالة تسويد الحواف لمنع الغش - يجب تطابق التدريب """
    img = np.array(pil_img)
    pad = 35 
    img[:pad, :, :] = 0      # فوق
    img[-pad:, :, :] = 0     # تحت
    img[:, :pad, :] = 0      # شمال
    img[:, -pad:, :] = 0     # يمين
    return Image.fromarray(img)

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
            nn.Linear(in_features, 1) 
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

# --- [تعديل جوهري] ربط الـ Preprocess بدالة الـ Blackout لمنع تشتت الـ Inference ---
img_size = chest_info["image_size"] # 224
chest_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),        
    transforms.Lambda(blackout_edges),           # خطوة التسويد الإجبارية
    transforms.CenterCrop((img_size, img_size)),  
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
        
        current_grads = gradients.detach()
        current_acts = activations.detach()
        
        pooled_gradients = torch.mean(current_grads, dim=[2, 3]).squeeze(0) 
        
        heatmap = torch.zeros(current_acts.shape[2:], dtype=torch.float32, device=device)
        for i in range(current_acts.size(1)):
            heatmap += pooled_gradients[i] * current_acts[0, i, :, :]
            
        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        # --- [تعديل جوهري] عتبة الـ Threshold لقص الغيامة الزرقاء وتحديد بؤرة المرض الحادة ---
        heatmap[heatmap < 0.5] = 0 
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        # --- تجهيز خلفية المعاينة بنفس خصائص الـ Preprocess لضمان تطابق الأبعاد ---
        # 1. نسوي الحواف
        processed_raw = blackout_edges(raw_image)
        img_cv = cv2.cvtColor(np.array(processed_raw), cv2.COLOR_RGB2BGR)
        # 2. ريسايز لـ 256
        img_cv = cv2.resize(img_cv, (256, 256))
        # 3. سنتر كروب لـ 224
        start_x = (256 - img_size) // 2
        start_y = (256 - img_size) // 2
        img_cv_cropped = img_cv[start_y:start_y+img_size, start_x:start_x+img_size]
        
        # تكبير وتلوين الـ Heatmap
        heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # دمج الـ Heatmap النظيف فوق الصورة المجهزة
        superimposed_img = cv2.addWeighted(img_cv_cropped, 0.6, heatmap_colored, 0.4, 0)
        
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