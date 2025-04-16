from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify a list of allowed origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# selected_features = ['Hours_Studied', 'Previous_Scores', 'Attendance', 'Sleep_Hours']
class ModelInput(BaseModel):
    Hours_Studied: float = Field(..., title="Hours Studied", description="Number of hours the student studied", le=24.0, ge=0.0)
    Previous_Scores: float = Field(..., title="Previous Scores", description="Previous scores of the student", le=100.0, ge=0.0)
    Attendance: int = Field(..., title="Attendance", description="Attendance percentage of the student", le=30, ge=0)
    Sleep_Hours: float = Field(..., title="Sleep Hours", description="Number of hours the student slept", le=24.0, ge=0.0)

# Load model and scaler
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def read_root():
    return {"this is root endpoint for StudentPerformance"}

@app.post("/predict")
def read_item(item: ModelInput):
    data = item.dict()
    data_array = np.array([[data['Hours_Studied'], data['Previous_Scores'], data['Attendance'], data['Sleep_Hours']]])
    scaled_data = scaler.transform(data_array)
    prediction = model.predict(scaled_data)
    return {"prediction": prediction.tolist()}
