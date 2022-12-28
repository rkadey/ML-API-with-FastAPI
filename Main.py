from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel


app = FastAPI(title ='Credit Card Churn Prediction', version = 1.0, description = 'Classification Machine Learning Prediction')
model = joblib.load("joblib_CC_Model.pkl")

class Data(BaseModel):



 @app.post("/predict")

 async def predict(data):
    prediction = model.predict()
    return prediction

if __name__ == '__main__':
    uvicorn.run("Main:app", reload = True)