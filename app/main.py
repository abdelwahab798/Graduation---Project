from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1. استيراد إعدادات قاعدة البيانات والجداول
from Database.config import engine, Base
import Database.models as models

# 2. استيراد الـ Routers (الـ Auth + الـ 6 موديلات بتوعك)
from api.routers import auth
from api.routers import chest_xray
from api.routers import Brain_Tumor
from api.routers import Brain_Stroke
from api.routers import Diabetic_Retinopathy
from api.routers import CKD
from api.routers import Diabetes
from api.routers import correction
from api.routers import history
from api.routers import RAG
from fastapi.staticfiles import StaticFiles



# 3. تعريف الـ FastAPI App بالـ Metadata الاحترافية بتاعتك
app = FastAPI(
    title="MediScan AI Platform API",
    description="Production-ready End-to-End Database and APIs for Medical Imaging Analysis (Grad-CAM Enabled)",
    version="1.0.0"
)

# 4. أمر خلق جداول الـ SQLite تلقائياً عند تشغيل السيرفر
Base.metadata.create_all(bind=engine)

# 5. إعدادات الـ CORS Middleware (مهمة جداً لربط الـ Front-end)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # أو حدد بورت الفرونت إند بتاعك
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], # 💡 مهم جداً عشان يستقبل الـ Authorization Header
)


# 6. تسجيل جميع الـ Routers في الـ App
app.include_router(auth.router) # راوتر الـ Auth لـ Login و Sign up
app.include_router(chest_xray.router)
app.include_router(Brain_Tumor.router)
app.include_router(Brain_Stroke.router)
app.include_router(Diabetic_Retinopathy.router)
app.include_router(CKD.router)
app.include_router(Diabetes.router)
app.include_router(correction.router)
app.include_router(history.router)
app.include_router(RAG.router)
app.mount("/static", StaticFiles(directory="Grad-cam"), name="static")

# 7. الـ Root Endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to MediScan AI Production API Server",
        "database_status": "Connected & Synchronized Successfully"
    }