from fastapi import FastAPI
from api.routers import chest_xray
from api.routers import Brain_Tumor
app = FastAPI(title="MediScan AI Platform")

app.include_router(chest_xray.router)
app.include_router(Brain_Tumor.router)

@app.get("/")
def home():
    return {"message": "Welcome to MediScan AI API"}