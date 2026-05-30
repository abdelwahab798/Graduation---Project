from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# استدعاء الـ Routers الخاصة بك
from api.routers import Brain_Stroke
from api.routers import Brain_Tumor
from api.routers import chest_xray
from api.routers.Diabetic_Retinopathy import router as diabetic_router

# 1. تعريف التطبيق مرة واحدة فقط باسم موحد
app = FastAPI(
    title="MediScan AI Platform API",
    description="Production API Server for Medical Imaging Analysis (Grad-CAM Enabled)",
    version="1.0.0"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
)

app.include_router(chest_xray.router)
app.include_router(Brain_Tumor.router)
app.include_router(Brain_Stroke.router)
app.include_router(diabetic_router)

@app.get("/")
def home():
    return {"message": "Welcome to MediScan AI Production API Server"}