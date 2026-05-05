from fastapi import FastAPI
from api.routers import chest_xray
app = FastAPI(title="MediScan AI Platform")

app.include_router(chest_xray.router)

@app.get("/")
def home():
    return {"message": "Welcome to MediScan AI API"}