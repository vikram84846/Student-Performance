from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
app = FastAPI()

# selected_features = ['Hours_Studied', 'Previous_Scores', 'Attendance', 'Sleep_Hours']
class ModelInput(BaseModel):
    Hours_Studied: float = Field(..., title="Hours Studied", description="Number of hours the student studied",le=0.0,ge=24.0)
    Previous_Scores: float = Field(..., title="Previous Scores", description="Previous scores of the student", le=0.0, ge=100.0)
    Attendance: int = Field(..., title="Attendance", description="Attendance percentage of the student", le=0, ge=30)
    Sleep_Hours: float = Field(..., title="Sleep Hours", description="Number of hours the student slept", le=0.0, ge=24.0)




model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.get("/")
def read_root():
    return{"this is root endpoint for StdentPerformance"}

@app.post("/predict")
def read_item(item: ModelInput):
    data = item.dict()
    # data = np.array([[data['Hours_Studied'], data['Previous_Scores'], data['Attendance'], data['Sleep_Hours']]])
    data = np.array([[data['Hours_Studied'], data['Previous_Scores'], data['Attendance'], data['Sleep_Hours']]])
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return {"prediction": prediction.tolist()}