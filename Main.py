from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector as selector
import pandas as pd
import numpy as np



app = FastAPI(title ='Credit Card Churn Prediction', version = 1.0, description = 'Classification Machine Learning Prediction')

class model_input(BaseModel):
    Customer_Age : int
    Gender : int
    Dependent_count : int
    Education_level : int
    Marital_status : str
    Income_category : str
    Card_category : str
    months_on_book : int
    Total_Relationship_Count : int
    Months_Inactive_12_mon : int
    Contacts_Count_12_mon : int
    Credit_Limit : int
    Total_Revolving_Bal : int
    Avg_Open_To_Buy : int
    Total_Amt_Chng_Q4_Q1 : int
    Total_Trans_Amt : int
    Total_Trans_Ct : int
    Total_Ct_Chng_Q4_Q1 : int
    Avg_Utilization_Ratio : int


  

# load the model
model = joblib.load("joblib_CC_Model.pkl")


  
@app.post("/credit_card_churn_prediction")
async def predicts(input:model_input):
    # Numeric Features
    num_features = [[input.Customer_Age, input.Dependent_count, input.months_on_book, input.Total_Relationship_Count, input.Months_Inactive_12_mon, 
                   input.Contacts_Count_12_mon, input.Credit_Limit, input.Total_Revolving_Bal, input.Avg_Open_To_Buy, input.Total_Amt_Chng_Q4_Q1, input.Total_Trans_Amt, 
                   input.Total_Trans_Ct, input.Total_Ct_Chng_Q4_Q1, input.Avg_Utilization_Ratio, input.Gender, input.Education_level]]
    num_pipeline = Pipeline([("Scaler", StandardScaler())])
    
    # Categorical Features
    cat_features = [[input.Gender, input.Education_level, input.Marital_status, input.Card_category]]
    cat_pipeline = Pipeline([("onehot", OneHotEncoder())])

    predict_input = ColumnTransformer(transformers = [("numeric_preprocessing", num_pipeline, num_features),
                                       ("categorical_preprocessing", cat_pipeline, cat_features)])

    final_input = np.array(predict_input.fit_transform([[model_input]]), dtype = np.str)

    prediction = model.predict(np.array([[final_input]]).reshape(1, 1))
    return prediction

if __name__ == '__main__':
    uvicorn.run("Main:app", reload = True)