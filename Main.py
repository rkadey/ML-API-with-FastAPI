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
    Gender : str
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
    num_features = [["Customer_Age", "Dependent_count", "months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", 
                   "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Trans_Amt", 
                   "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]]
    num_pipeline = Pipeline([("Scaler", StandardScaler())])
    
    # Categorical Features
    cat_features = [["Gender", "Education_level", "Marital_status", "Card_category"]]
    cat_pipeline = Pipeline([("onehot", OneHotEncoder())])

    predict_input = ColumnTransformer(transformers = [("numeric_preprocessing", num_pipeline, num_features),
                                       ("categorical_preprocessing", cat_pipeline, cat_features)])
    #print(model_input)
    df = pd.DataFrame([input])
    final_input = np.array(predict_input.fit_transform(df), dtype = np.str)

    prediction = model.predict(np.array([[final_input]]).reshape(1, 1))
    return prediction

if __name__ == '__main__':
    uvicorn.run("Main:app", reload = True)