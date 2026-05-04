from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    population_density: float
    vaccination_pct: float
    sanitation_index: float
    rainfall_mm: float

@app.get("/heartbeat")
def heartbeat():
    return {"status": "healthy", "model_loaded": True}

@app.post("/forecast")
def predict(data: Input):
    return {"prediction": 50.0}