from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ===== Database Setup =====
from database.database import engine
from database import models
models.Base.metadata.create_all(bind=engine)  # بينشئ الجداول أول ما السيرفر يشتغل

# ===== Routers =====
from api.routers import Brain_Stroke
from api.routers import Brain_Tumor
from api.routers import chest_xray
from api.routers import Diabetic_Retinopathy
from api.routers import CKD
from api.routers import Diabetes

app = FastAPI(
    title="MediScan AI Platform API",
    description="Production API Server for Medical Imaging Analysis (Grad-CAM Enabled)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chest_xray.router)
app.include_router(Brain_Tumor.router)
app.include_router(Brain_Stroke.router)
app.include_router(Diabetic_Retinopathy.router)
app.include_router(CKD.router)
app.include_router(Diabetes.router)

@app.get("/")
def home():
    return {"message": "Welcome to MediScan AI Production API Server"}
