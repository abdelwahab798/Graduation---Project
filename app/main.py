from fastapi import FastAPI
from api.routers import Brain_Stroke
from api.routers import Brain_Tumor
from api.routers import chest_xray

app = FastAPI(
    title="MediScan AI Platform API",
    version="1.0.0"
)
app.include_router(chest_xray.router)
app.include_router(Brain_Tumor.router)
app.include_router(Brain_Stroke.router)

@app.get("/")
def home():
    return {"message": "Welcome to MediScan AI Production API Server"}